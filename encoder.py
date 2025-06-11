# ბიბლიოთეკები
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    import cv2

# CIFAR-10
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

# სურათების "დამახარვეზებელი" კოდი
    def add_defects(images):
        def apply_defects(img):
            img = (img * 255).astype(np.uint8).copy()
    
            # დაბინდვა
            h, w, _ = img.shape
            patch_size = np.random.randint(8, 16)
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            patch = img[y:y+patch_size, x:x+patch_size]
            blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
            img[y:y+patch_size, x:x+patch_size] = blurred_patch
    
            # თეთრი ხაზის დატანა
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = (255, 255, 255)
            thickness = np.random.randint(1, 3)
            img = cv2.line(img, pt1, pt2, color, thickness)
    
            return img.astype(np.float32) / 255.0
    
        return np.array([apply_defects(img) for img in images])

# "დამახარვეზებელი" კოდის შემოწმება
    x_clean_sample = x_train[:5]
    x_defected_sample = add_defects(x_clean_sample)

    def show_clean_vs_defected(clean, defected, n=5):
        plt.figure(figsize=(12, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(clean[i])
            plt.title("Original")
            plt.axis("off")
    
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(defected[i])
            plt.title("Defected")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    
    
    show_clean_vs_defected(x_clean_sample, x_defected_sample)

# ავტოენკოდერის აგება
    def build_autoencoder():
        input_img = layers.Input(shape=(32, 32, 3))
    
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
        return models.Model(input_img, decoded)

# დანაკარგის კომბინირებული ფუნქცია
    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return 0.5 * mse + 0.5 * (1 - ssim)

# კომპილაცია
    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer='adam', loss=combined_loss)
    autoencoder.summary()

# 10000:3000 გასაწვრთნი და სატესტო სურათის შერჩევა CIFAR-10-დან და მათი დამახინჯება
    x_train_small = x_train[:10000]
    x_test_small = x_test[:3000]
    
    x_train_defected_small = add_defects(x_train_small)
    x_test_defected_small = add_defects(x_test_small)

# ავტოენკოდერის გაწვრთნა
    autoencoder.fit(
        x_train_defected_small, x_train_small,
        epochs=15,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_defected_small, x_test_small)
    )

# ვიზუალიზაცია
decoded_imgs = autoencoder.predict(x_test_defected_small[:10])

def show_reconstruction(original, defected, reconstructed, n=10):
    plt.figure(figsize=(18, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(defected[i])
        plt.title("Defected")
        plt.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

show_reconstruction(
    x_test_small[:10],
    x_test_defected_small[:10],
    decoded_imgs
)
