TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "yolov8n",
        "backbone_pretrained": "",
    },

    "yolo": {
        "classes": 13,
    },

    "batch_size": 1,

    "confidence_threshold": 0.5,
    "nms_threshold": 0.45,

    "images_path": "dataset/test/images/",
    "classes_names_path": "classes.txt",

    "img_h": 640,
    "img_w": 640,

    "parallels": [0],
    "pretrain_snapshot": "",
}