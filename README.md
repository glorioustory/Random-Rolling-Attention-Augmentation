This code improves the self-attention computation unit AATtn in the A2C2f module of YOLOv12 by performing a random rolling operation on the three tensors q/k/v before computation.
While maintaining the number of model parameters and computational cost, it improves metrics such as map50 in IP102 pest and disease detection and tomato leaf disease detection.
