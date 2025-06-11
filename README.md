# მეორე შუალედური გამოცდა

ავტოენკოდერი გაწვრთნილია CIFAR-10 ბაზაზე. ამოღებულია ყველა სურათი (X-მნიშვნელობა). შემდეგ, სპეციალური კოდის მეშვეობით ისინი დამახანჯებული არიან. ავტოენკოდერის მიზანია დამახინჯებული სურათების მიღების შემდეგ მოგვცეს მათი აღდგენილი ვერსიები.

ამ ფაილში განხილული იქნება სამუშაოს ყველა ნაბიჯი და კოდის გარკვეული ნაწილები. სრული კოდის ნახვა კი შესაძლებელია აქ ![https://github.com/Nodar-Melkonyan/midtermII/blob/main/encoder.py]

## ნაბიჯი 1
ყველა საჭირო ბიბლიოთეკის გადმოწერა

    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    import cv2

## ნაბიჯი 2
CIFAR-10 ბაზის გამოწერა

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

აღსანიშნავია, რომ y-მნიშვნელობები არ მიმითითებია. იმის გამო, რომ ავტოენკოდერის მიზანია სურათების პირვანდელი სახის აღდგენა და არა კლასიფიკაცია, მიზანშეწონილად ჩავთვალე არ დამემატებინა y-მნიშვნელობები (labels).

## ნაბიჯი 3
ფუნქცია, რომელიც დაახარვეზებს სურათებს დაბინდვის და თეთრი ხაზების დატანის მეშვეობით

    def add_defects(images)

## ნაბიჯი 4
ავტოენკოდერის შექმნამდე დავრწმუნდეთ, რომ დახარვეზები ფუნქცია მუშაობს

ამისათვის ავიღოთ 5 სურათი ბაზიდან

    x_clean_sample = x_train[:5]

დავახარვეზოთ ისინი

    x_defected_sample = add_defects(x_clean_sample)

შევქმნათ ფუნქცია, რომელიც matplotlib-ის გამოყენებით გამოიტანს ამ 5 სურათის შედარებით წყვილებს

    def show_clean_vs_defected(clean, defected, n=5)

გავუშვათ იგი ჩვენი 5 წყვილი სურათისთვის

    show_clean_vs_defected(x_clean_sample, x_defected_sample)

და ვნახოთ, რომ სურათები ნამდვილად ხარვეზდება !დახარვეზებული და დედანი სურათები[https://github.com/Nodar-Melkonyan/midtermII/blob/main/Orig-def.png]

## ნაბიჯი 5
ავტოენკოდერის აგება

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

## ნაბიჯი 6
კომპილირების ეტაპი

აღსანიშნავია, რომ თავდაპირველად გამოყენებული მქონდა საშუალო კვადრატული შეცდომა (Mean Squarred Error) დახარვეზებულ და დედან სურათებს შორის სხვაობების დასაჭერად. მიუხედავად იმისა, რომ ავტოენკოდერმა წარმატებით შეასრულა თავისი ამოცანა და დახარვეზებულ სურათებს მოაშორა თეთრი ხაზები, აღდგენილი სურათები ძალიან დაბინდული აღმოჩნდა.

ამ ხარვეზის გამოსასწორებლად შევიტანე სტრუქტურული მსგავსების ინდექსიც (SSIM).

შესაბამისად, ჯერ შევქმენი ამ ორი პარამეტრის შემცველი კომპინირებული ფუნქცია

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return 0.5 * mse + 0.5 * (1 - ssim)

და დავაკომპილირე

    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer='adam', loss=combined_loss)
    autoencoder.summary()

## ნაბიჯი 7
მონაცემთა ნაკრების განსაზღვრა

    x_train_small = x_train[:10000]
    x_test_small = x_test[:3000]
    
    x_train_defected_small = add_defects(x_train_small)
    x_test_defected_small = add_defects(x_test_small)

ხაზგასასმელია, რომ პირველი გაშვების დროს აღებული მქონდა 5000:1000 სურათი. თუმცა, გაუმჯობესების მიზნით ეგ შეფარდება გავასამმაგე — 15000:3000. იმისი გამო, რომ ამდენი სურათის დამუშავებას დიდი დრო დასჭირდა, გასაწვრთნი სურათების რიცხვი 10000-ამდე დავიყვანე.

## ნაბიჯი 8
გაწვრთნა 15 ეპოქად

    autoencoder.fit(
        x_train_defected_small, x_train_small,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_defected_small, x_test_small)
    )

აღსანიშნავია, რომ პირველი გაშვების დროს (10 ეპოქად) დანაკარგმა 71% შეადგინა. ოპტიმიზირებული ავტოენკოდერის შემთხვევაში ეს მაჩვენებელი გაცილებით დაბალია.!მეორე ავტოენკოდერის დანაკარგი 13,63% პროცენტი არის. !დანაკარგი[https://github.com/Nodar-Melkonyan/midtermII/blob/main/Loss.png]
უკეთესი შედეგის მიღწევაც შეიძლებოდა, თუმცა შეზღუდული დროის გამო ეს ვერ მოხერხდა ამ მომენტში.

## ნაბიჯი 9
ბოლოს კი ვიზუალიზაცია მოვახდინე შემდეგი ფუნქციის გამოყენებით

    def show_reconstruction(original, defected, reconstructed, n=10)

მისი გამოძახებით ვხედავთ სამ რიგ სურათს: პირველში — დედანს, მეორეში — დახარვეზებულს, მესამეში — აღდგენილს.

    show_reconstruction(
        x_test_small[:10],
        x_test_defected_small[:10],
        decoded_imgs
    )
სადაც decoded_imgs არის ავტოენკოდერში გატარებული 10 სურათი CIFAR-იდან

    decoded_imgs = autoencoder.predict(x_test_defected_small[:10])

!მიღებული შედეგების მაგალითი[https://github.com/Nodar-Melkonyan/midtermII/blob/main/Final%20Result.png]
ჩანს, რომ ავტოენკოდერმა სწორად მოაშორა თეთრი ხაზები და აღადგინა სურათი. თუმცა, დარჩა გარკვეული დაბინდულობა, რისი გამოსწორება შეიძლება ავტოენკოდერის ოპტიმიზაციით. ამას კი სჭირდება მეტი რესურსი.
შედარებისთვის, ვურთავ პირველი ავტოენკოდერის შედეგსაც !პირველი შედეგი[https://github.com/Nodar-Melkonyan/midtermII/blob/main/First%20Encoder.png]
