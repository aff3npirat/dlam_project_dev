import timm


_resnet50 = timm.create_model("resnet50", pretrained=False)
