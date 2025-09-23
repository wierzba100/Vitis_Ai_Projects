TRAINING_PARAMS = {
    # Parametry modelu (dla zgodności zostawiam klucz, choć YOLOv8 nie potrzebuje anchorów/backbone)
    "model_params": {
        "backbone_name": "yolov8n",
        "backbone_pretrained": "",
    },

    # Informacje o liczbie klas
    "yolo": {
        "classes": 13,   # zgodnie z data.yaml
    },

    # Batch size – możesz ustawić wedle potrzeb przy treningu/inferencji
    "batch_size": 1,

    # Progi detekcji
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45,

    # Ścieżki do danych i nazw klas
    "images_path": "dataset/test/images/",       # z Twojego data.yaml
    "classes_names_path": "classes.txt",         # plik tekstowy z nazwami klas

    # Rozdzielczość wejściowa (YOLOv8n standardowo: 640x640, ale możesz zmienić)
    "img_h": 640,
    "img_w": 640,

    # Pozostałe (opcjonalne)
    "parallels": [0],
    "pretrain_snapshot": "",   # brak, bo model już jest skompilowany do DPU
}