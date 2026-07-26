def get_pretrained_checkpoint_url():
    base = "https://example.invalid/assets"
    checkpoints = {
        "a": {
            "url": f"{base}/hidden-audio-model.pth",
            "filename": "hidden-audio-model.pth",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        },
        "b": {
            "url": f"{base}/hidden-audio-model-b.pth",
            "filename": "hidden-audio-model-b.pth",
            "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        },
    }
    return checkpoints["a"]["url"]
