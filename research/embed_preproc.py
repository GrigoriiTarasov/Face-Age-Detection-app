def img_to_input(path, target_size=(224, 224)):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = utils.preprocess_input(x, version=2)
    x = normalize_input(x, normalization="VGGFace")

    return x


def img_to_input(paths, target_size=(224, 224)):
    batch_images = []
    for path in paths:
        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)
        x = normalize_input(x, normalization="VGGFace")
        batch_images.append(x)

    batch_images = np.concatenate(batch_images, axis=0)
    return batch_images
