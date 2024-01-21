import argparse
import base64
import json
import os.path
import subprocess
import traceback
import soundfile

import hay_say_common as hsc
import jsonschema
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError

ARCHITECTURE_NAME = 'styletts_2'
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
INPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'input')
OUTPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'output')
WEIGHTS_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'Models', 'LJSpeech')
TEMP_FILE_EXTENSION = '.wav'

PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, '.venvs', ARCHITECTURE_NAME, 'bin', 'python')
INFERENCE_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'command_line_interface.py')

WEIGHTS_FILE_EXTENSION = '.pth'

# todo: update get_single_file_with_extension in hay_say_common so I can just specify a tuple of possible extensions
CONFIG_FILE_EXTENSION = '.yml'
CONFIG_FILE_EXTENSION_ALT = '.yaml'

STYLE_FILE_EXTENSION = '.json'

app = Flask(__name__)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            (user_text, character, noise, style_blend, diffusion_steps, embedding_scale, use_long_form,
             input_hash, enable_reference_audio, timbre_ref_blend, prosody_ref_blend,
             output_filename_sans_extension, gpu_id, session_id) = parse_inputs()
            reference_audio = prepare_reference_audio(input_hash, enable_reference_audio, cache,
                                                      session_id)
            execute_program(user_text, character, noise, style_blend, diffusion_steps, embedding_scale, use_long_form,
                            reference_audio, enable_reference_audio, timbre_ref_blend, prosody_ref_blend,
                            output_filename_sans_extension, gpu_id)
            copy_output(output_filename_sans_extension, session_id)
            hsc.clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = hsc.construct_full_error_message('No input files to report', get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE)

    def parse_inputs():
        schema = {
            'type': 'object',
            'properties': {
                'Inputs': {
                    'type': 'object',
                    'properties': {
                        'User Text': {'type': 'string'},
                        'User Audio': {'type': ['string', 'null']}
                    },
                    'required': ['User Text']
                },
                'Options': {
                    'type': 'object',
                    'properties': {
                        'Character': {'type': 'string'},
                        'Noise': {'type': 'number'},
                        'Style Blend': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'Diffusion Steps': {'type': 'integer', 'minimum': 1},
                        'Embedding Scale': {'type': 'number'},
                        'Use Long Form': {'type': 'boolean'},
                        'Enable Reference Audio': {'type': 'boolean'},
                        'Timbre Reference Blend': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'Prosody Reference Blend': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    },
                    'required': ['Character', 'Noise', 'Style Blend', 'Diffusion Steps', 'Embedding Scale',
                                 'Use Long Form']
                },
                'Output File': {'type': 'string'},
                'GPU ID': {'type': ['string', 'integer']},
                'Session ID' : {'type': ['string', 'null']}
            },
            'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
        }

        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        user_text = request.json['Inputs']['User Text']
        character = request.json['Options']['Character']
        noise = request.json['Options']['Noise']
        style_blend = request.json['Options']['Style Blend']
        diffusion_steps = request.json['Options']['Diffusion Steps']
        embedding_scale = request.json['Options']['Embedding Scale']
        use_long_form = request.json['Options']['Use Long Form']
        input_hash = request.json['Inputs']['User Audio']
        enable_reference_audio = request.json['Options']['Enable Reference Audio']
        timbre_ref_blend = request.json['Options']['Timbre Reference Blend']
        prosody_ref_blend = request.json['Options']['Prosody Reference Blend']
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return (user_text, character, noise, style_blend, diffusion_steps, embedding_scale, use_long_form,
                input_hash, enable_reference_audio, timbre_ref_blend, prosody_ref_blend,
                output_filename_sans_extension, gpu_id, session_id)


    class BadInputException(Exception):
        pass


    def prepare_reference_audio(input_hash, enable_reference_audio, cache, session_id):
        """Temporarily pull the reference file out of the cache and save it to a file."""
        if not enable_reference_audio or input_hash is None:
            return None
        target = os.path.join(INPUT_COPY_FOLDER, input_hash + TEMP_FILE_EXTENSION)
        try:
            array, samplerate = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_hash)
            soundfile.write(target, array, samplerate)
        except Exception as e:
            raise Exception("Unable to save reference audio to a temporary file.") \
                from e
        return target

    def get_config_file(character):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        try:
            config_file = hsc.get_single_file_with_extension(character_dir, CONFIG_FILE_EXTENSION)
        except Exception:
            config_file = hsc.get_single_file_with_extension(character_dir, CONFIG_FILE_EXTENSION_ALT)
        return config_file

    def get_style_file(character):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        style_file = hsc.get_files_with_extension(character_dir, STYLE_FILE_EXTENSION)
        return style_file[0] if style_file else None

    def execute_program(user_text, character, noise, style_blend, diffusion_steps, embedding_scale, use_long_form,
                        reference_audio, enable_reference_audio, timbre_ref_blend, prosody_ref_blend,
                        output_filename_sans_extension, gpu_id):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        config_file = get_config_file(character)
        style_file = get_style_file(character)
        arguments = [
            '--text', user_text,
            '--weights_file', hsc.get_single_file_with_extension(character_dir, WEIGHTS_FILE_EXTENSION),
            '--config_file', config_file,
            '--output_filepath', os.path.join(OUTPUT_COPY_FOLDER, output_filename_sans_extension + TEMP_FILE_EXTENSION),
            # Optional Parameters
            *(['--noise', str(noise)] if noise is not None else [None, None]),
            *(['--style_blend', str(style_blend)] if style_blend is not None else [None, None]),
            *(['--diffusion_steps', str(diffusion_steps)] if diffusion_steps is not None else [None, None]),
            *(['--embedding_scale', str(embedding_scale)] if embedding_scale is not None else [None, None]),
            *(['--embedding_scale', str(embedding_scale)] if embedding_scale is not None else [None, None]),
            *(['--use_long_form'] if use_long_form else [None]),
            *(['--reference_audio', reference_audio] if enable_reference_audio else [None, None]),
            *(['--reference_style', style_file] if style_file and not enable_reference_audio else [None, None]),
            *(['--timbre_ref_blend', str(timbre_ref_blend)] if timbre_ref_blend is not None else [None, None]),
            *(['--prosody_ref_blend', str(prosody_ref_blend)] if prosody_ref_blend is not None else [None, None]),
        ]
        arguments = [argument for argument in arguments if argument]  # Removes all "None" objects in the list.
        env = hsc.select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, INFERENCE_SCRIPT_PATH, *arguments], env=env)


    def copy_output(output_filename_sans_extension, session_id):
        array_output, sr_output = hsc.read_audio(os.path.join(OUTPUT_COPY_FOLDER,
                                                          output_filename_sans_extension + TEMP_FILE_EXTENSION))
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)


    def get_temp_files():
        return [os.path.join(OUTPUT_COPY_FOLDER, file) for file in os.listdir(OUTPUT_COPY_FOLDER)] + \
               [os.path.join(INPUT_COPY_FOLDER, file) for file in os.listdir(INPUT_COPY_FOLDER)]


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py', description='A webservice interface for voice conversion with RVC')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6578)
