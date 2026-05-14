import os

def patch_file(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found, skipping patch.")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN and AutoConfig
    search_import = "from llava.constants import DEFAULT_IMAGE_TOKEN\n"
    replace_import = (
        "from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN\n"
        "from transformers import AutoConfig\n"
    )
    if search_import in content:
        content = content.replace(search_import, replace_import)

    # Initialize self.mm_use_im_start_end in ModelWorker.__init__
    search_init = """        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:"""
    replace_init = """        if model_path.endswith("/"):
            model_path = model_path[:-1]

        self.mm_use_im_start_end = False
        try:
            config = AutoConfig.from_pretrained(model_path)
            self.mm_use_im_start_end = getattr(config, 'mm_use_im_start_end', False)
        except Exception as e:
            logger.warning(f"Failed to load model config from {model_path}: {e}. Disabling image start/end tokens.")

        if model_name is None:"""
    if search_init in content:
        content = content.replace(search_init, replace_init)

    # Apply replacement of DEFAULT_IMAGE_TOKEN in generate_stream
    search_replace = """                # FIXME: for image-start/end token
                # replace_token = DEFAULT_IMAGE_TOKEN
                # if getattr(self.model.config, 'mm_use_im_start_end', False):
                #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                # prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                prompt = prompt.replace(' ' + DEFAULT_IMAGE_TOKEN + '\\n', DEFAULT_IMAGE_TOKEN)
                prompt_split = prompt.split(DEFAULT_IMAGE_TOKEN)"""

    replace_replace = """                prompt = prompt.replace(' ' + DEFAULT_IMAGE_TOKEN + '\\n', DEFAULT_IMAGE_TOKEN)

                replace_token = DEFAULT_IMAGE_TOKEN
                if self.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                prompt_split = prompt.split(DEFAULT_IMAGE_TOKEN)"""
    if search_replace in content:
        content = content.replace(search_replace, replace_replace)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MagicQuill', 'LLaVA', 'llava', 'serve', 'sglang_worker.py')
    patch_file(target)
