import customtkinter as ctk
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import scrolledtext
from tkinter import font
import threading
import sys
import shutil
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  roc_auc_score, confusion_matrix, classification_report
import random
import tensorflow as tf
from keras.models import Model
from keras import layers
import ast
import re


"""print(f"Version de Python : {sys.version}\n")

try:
    import tensorflow as tf
    print(f"TensorFlow : {tf.__version__}")
except ImportError:
    print("TensorFlow : Non installé")

try:
    import keras
    print(f"Keras : {keras.__version__}")
except ImportError:
    print("Keras : Non installé")

try:
    import numpy as np
    print(f"NumPy : {np.__version__}")
except ImportError:
    print("NumPy : Non installé")

try:
    import matplotlib
    print(f"Matplotlib : {matplotlib.__version__}")
except ImportError:
    print("Matplotlib : Non installé")

try:
    import seaborn as sns
    print(f"Seaborn : {sns.__version__}")
except ImportError:
    print("Seaborn : Non installé")

try:
    import sklearn
    print(f"Scikit-learn : {sklearn.__version__}")
except ImportError:
    print("Scikit-learn : Non installé")

try:
    from PIL import Image
    print(f"Pillow : {Image.__version__}")
except ImportError:
    print("Pillow : Non installé")

try:
    import customtkinter
    print(f"CustomTkinter : {customtkinter.__version__}")
except ImportError:
    print("CustomTkinter : Non installé")

try:
    import auto_py_to_exe
    print(f"auto-py-to-exe : {auto_py_to_exe.__version__}")
except ImportError:
    print("auto-py-to-exe : Non installé")"""

global_export_folder_pathAUGMENT = None  # Define a global variable
def resource_path(relative_path):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

###################################################################################################################################################

def remove_exif_from_jpg(input_dir, output_dir):

    print("EXIF metadata removal in progress…")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all files and directories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directories in the output directory
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Process each file in the current directory
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(root, filename)
                output_file_path = os.path.join(output_path, filename)

                try:
                    with Image.open(input_path) as img:
                        # Save the image without EXIF data while keeping the original extension
                        img.save(output_file_path, format=img.format)
                    #print(f'Processed {input_path} to {output_file_path}')
                except Exception as e:
                    print(f'Error processing {input_path}: {e}')

    # Obtenir les noms des sous-dossiers dans UserImages
    ###class_names = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    # Créer un dictionnaire {i: class_name} où i est un entier à partir de 0 et class_name est le nom du sous-dossier
    ###new_dict = {i: class_name for i, class_name in enumerate(class_names)}
    # Définir le nom du fichier en fonction du nombre de classes
    ###dict_name = 'your_dictionary_with' + str(len(class_names)) + 'classes.txt'
    # Définir le chemin complet pour sauvegarder le fichier
    ###dict_path = os.path.join(output_dir, dict_name)  # Sauvegarder le dictionnaire dans le fichier
    ###with open(dict_path, 'w') as file:
    ###    file.write(str(new_dict))
    print("EXIF metadata removal completed. You can now proceed to annotation and sorting on the next page => Annotate and sort")

#############################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa###################################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

def create_and_augment(input, output, mode, augment_training, augment_validation, augment_test):

    import keras
    # Define paths
    UserImages = input
    outputfolder = output
    entrainement = os.path.join(outputfolder, "training")
    validation = os.path.join(outputfolder, "validation")
    test = os.path.join(outputfolder, "test")

    fill_mode_choosen = mode #"constant", "reflect", "wrap", "nearest"

    # Obtenir les noms des sous-dossiers dans UserImages
    ###class_names = [name for name in os.listdir(UserImages) if os.path.isdir(os.path.join(UserImages, name))]
    # Créer un dictionnaire {i: class_name} où i est un entier à partir de 0 et class_name est le nom du sous-dossier
    ###new_dict = {i: class_name for i, class_name in enumerate(class_names)}
    # Définir le nom du fichier en fonction du nombre de classes
    ###dict_name = 'your_dictionary_with' + str(len(class_names)) + 'classes.txt'
    # Définir le chemin complet pour sauvegarder le fichier
    ###dict_path = os.path.join(outputfolder, dict_name)
    # Sauvegarder le dictionnaire dans le fichier
    ###with open(dict_path, 'w') as file:
    ###    file.write(str(new_dict))

    # Create target directories if they don't exist
    [os.makedirs(folder, exist_ok=True) for folder in [entrainement, validation, test]]


    # Walk through all files and directories in the UserImages directory
    all_files = []
    for root, dirs, files in os.walk(UserImages):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    # Shuffle the list of files
    random.shuffle(all_files)

    # Split the list into three parts
    split1 = int(len(all_files) * 0.8)
    split2 = int(len(all_files) * 0.9)

    # Separate the files into three lists
    files1 = all_files[:split1]
    files2 = all_files[split1:split2]
    files3 = all_files[split2:]

    # Function to move files while preserving folder structure
    def copy_files(files, destination_folder, source_root):
        for file_path in files:
            # Determine the relative path of the file
            relative_path = os.path.relpath(file_path, source_root)
            # Determine the target path
            target_path = os.path.join(destination_folder, relative_path)
            # Create any necessary directories in the target path
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # Vérifier si le fichier est une image avec des métadonnées à supprimer
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    # Charger l'image avec Pillow
                    with Image.open(file_path) as img:
                        # Enregistrer l'image sans métadonnées EXIF
                        img.save(target_path, format=img.format)
                        #print(f"Image sans métadonnées copiée : {file_path} -> {target_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {file_path}: {e}")

    # Move the files to their respective folders
    print("Training, validation, and test datasets creation in progress...")
    copy_files(files1, entrainement, UserImages)
    copy_files(files2, validation, UserImages)
    copy_files(files3, test, UserImages)

    def compter_elements_sous_dossiers(chemin):
        try:
            # Initialiser une liste pour stocker le nombre d'éléments
            resultats = []

            # Lister les éléments dans le répertoire
            liste_dossiers = os.listdir(chemin)

            for sous_dossier in liste_dossiers:
                chemin_complet = os.path.join(chemin, sous_dossier)

                # Vérifier si l'élément est un dossier
                if os.path.isdir(chemin_complet):
                    # Filtrer les fichiers selon les extensions d'image
                    fichiers_images = [
                        f for f in os.listdir(chemin_complet)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                    ]
                    # Compter uniquement les fichiers d'image filtrés
                    nombre_elements = len(fichiers_images)
                    resultats.append(nombre_elements)

            return resultats
        except FileNotFoundError:
            return "Le chemin spécifié n'existe pas."
        except NotADirectoryError:
            return "Le chemin spécifié n'est pas un dossier."


    # Exemple d'utilisation
    liste_nb_elements_entrainement = compter_elements_sous_dossiers(entrainement)
    liste_nb_elements_validation = compter_elements_sous_dossiers(validation)
    liste_nb_elements_test = compter_elements_sous_dossiers(test)

    max_element_entrainement = max(liste_nb_elements_entrainement)
    max_element_validation = max(liste_nb_elements_validation)
    max_element_test = max(liste_nb_elements_test)

    print(liste_nb_elements_entrainement, liste_nb_elements_validation, liste_nb_elements_test)  # Par exemple, [5, 3, 6, 4]
    print(max_element_entrainement, max_element_validation, max_element_test)

    #########################
    print("fill_mode_choosen", fill_mode_choosen)
    # Define augmentation layers
    if fill_mode_choosen:
        img_augmentation_layers = [
            keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=None),
            keras.layers.RandomRotation(factor=0.05, seed=None, fill_mode=fill_mode_choosen),
            keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=None, fill_mode=fill_mode_choosen),
        ]

    # Define augmentation function
    def img_augmentation(images):
        for layer in img_augmentation_layers:
            images = layer(images)
        return images


    # Function to augment images in a folder
    def augmenter_images_dossier(chemin_dossier, max_elements):
        try:
            # Lister les sous-dossiers
            sous_dossiers = os.listdir(chemin_dossier)

            for sous_dossier in sous_dossiers:
                chemin_complet = os.path.join(chemin_dossier, sous_dossier)

                if os.path.isdir(chemin_complet):
                    # Compter le nombre d'éléments dans le sous-dossier
                    fichiers_images = [
                        f for f in os.listdir(chemin_complet)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                    ]
                    nb_elements = len(fichiers_images)
                    images_to_augment = max_elements - nb_elements

                    if images_to_augment > 0:
                        num_images = len(fichiers_images)
                        duplicates_per_image = images_to_augment // num_images
                        remainder = images_to_augment % num_images

                        augmentation_count = 1

                        # Cycle through images for duplication
                        for file_index, original_image_name in enumerate(fichiers_images):
                            original_image_path = os.path.join(chemin_complet, original_image_name)

                            # Charger l'image
                            image = keras.utils.load_img(original_image_path)
                            input_arr = keras.utils.img_to_array(image)
                            input_arr = input_arr.reshape((1, *input_arr.shape))  # Ajouter la dimension de batch

                            # Augmenter l'image le nombre requis de fois
                            for i in range(duplicates_per_image + (1 if file_index < remainder else 0)):
                                # Appliquer l'augmentation
                                augmented_image_arr = img_augmentation(input_arr)
                                augmented_image = keras.utils.array_to_img(augmented_image_arr[0])

                                # Sauvegarder l'image augmentée avec un nouveau nom
                                base_name, ext = os.path.splitext(original_image_name)
                                new_image_name = f"{base_name}_aug{augmentation_count}{ext}"
                                new_image_path = os.path.join(chemin_complet, new_image_name)
                                augmented_image.save(new_image_path)

                                augmentation_count += 1
                                #print(augmentation_count)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")

    print("test: ", augment_test)
    print("validation: ", augment_validation)
    print("training ", augment_training)


    # Conditionally apply augmentation based on checkbox selections
    if augment_training:
        augmenter_images_dossier(entrainement, max_element_entrainement)
    if augment_validation:
        augmenter_images_dossier(validation, max_element_validation)
    if augment_test:
        augmenter_images_dossier(test, max_element_test)


    #######verification augmentation

    liste_nb_elements_entrainement = compter_elements_sous_dossiers(entrainement)
    liste_nb_elements_validation = compter_elements_sous_dossiers(validation)
    liste_nb_elements_test = compter_elements_sous_dossiers(test)

    print(liste_nb_elements_entrainement, liste_nb_elements_validation, liste_nb_elements_test)
    print("Training, validation, and test datasets have been created. You can now proceed to the next page => Train your model")

#############################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa###################################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

def train_model(yoursubject, yournumberofepochs, training_folder, validation_folder, save_dir, yourmodel_name, yourheight, yourwidth, yourpatience, yourstop_patience, yourlearningrate, yourfactor, yourbatch_size, yoursizeofadditionallayer1, yoursizeofadditionallayer2, yoursizeofadditionallayer3, YourDropout, Useflattenlayer, yourfreeze):
    yoursubject = yoursubject  # set the name of your subject
    yournumberofepochs = yournumberofepochs  # set your number of epochs for each trained model (usually <10 to identified your best parameters, and
    # between 20-30 to train your best models)
    yourpatience = int(yourpatience)  # 1 by default. Number of epochs to wait to adjust lr if monitored value does not improve
    yourstop_patience = int(yourstop_patience)  # 3 by default. Number of epochs to wait before stopping training if monitored value does not improve
    #yourthreshold = float(0.9)   0.9 by default. If train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
    yourlearningrate = float(yourlearningrate)  # .00001 by default. The initial learning of your model
    yourfactor = float(yourfactor)  # 0.5by default. Factor to reduce lr by
    yourfreeze = yourfreeze  # False by default. If true free weights of  the base model
    yourheight = int(yourheight)  # usually 224 in vgg models and predictive models used on ImageNet (an important dataset used to train models during competitions)
    yourwidth = int(yourwidth)  # usually 224 in vgg models and predictive models used on ImageNet  (an important dataset used to train models during competitions)
    yourmodel_name = yourmodel_name  # 'ResNet152V2', 'VGG19', 'ResNet101V2', 'ResNet50V2', 'MobileNetV2', 'InceptionV3',
    # 'VGG16', 'Xception','DenseNet121', 'DenseNet169', 'NASNetMobile', 'DenseNet201', 'NASNetLarge', '', ''
    yourbatch_size = int(yourbatch_size)  # better if you have more than 10 pictures by classe. So if you have 10 classes you can
    # set this parameter to 100. However increasing this parameter increase the memory needed
    yoursizeofadditionallayer1 = int(yoursizeofadditionallayer1)  # 8 - 4096... often beween 100 to 300

    print(f'YourDropout: {YourDropout}')
    print(f'Useflattenlayer: {Useflattenlayer}')
    print(f'yourfreeze: {yourfreeze}')
    print(f'yoursizeofadditionallayer1: {yoursizeofadditionallayer1}')
    print(f'yoursizeofadditionallayer2: {yoursizeofadditionallayer2}')
    print(f'yoursizeofadditionallayer3: {yoursizeofadditionallayer3}')

    training_folder = training_folder # set the location of your training files
    validation_folder = validation_folder # set the location of your validation files
    save_dir = save_dir  # set the location where to save your model

    channels = 3
    batch_size = yourbatch_size
    img_shape = (yourheight, yourwidth, channels)
    img_size = (yourheight, yourwidth)


    # Loading the datasets using image_dataset_from_directory
    ds_train = keras.utils.image_dataset_from_directory(
        training_folder,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=yourbatch_size,
        image_size=img_size,
        shuffle=True,
        seed=123,
        validation_split=None,
        subset=None,
    )
    ds_valid = keras.utils.image_dataset_from_directory(
        validation_folder,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=yourbatch_size,
        image_size=img_size,
        shuffle=True,
        seed=123,
        validation_split=None,
        subset=None,
    )



    # Get the number of different labels (classes)
    class_names = ds_train.class_names # Get class names
    NUM_CLASSES = len(class_names)
    print(f'Number of different labels (classes): {NUM_CLASSES}')

    # Créer un dictionnaire {i: class_name} où i est un entier à partir de 0 et class_name est le nom du sous-dossier
    new_dict = {i: class_name for i, class_name in
                enumerate(class_names)}  # Create a dictionary {integer: class_name}
    classes = list(new_dict.values())  # List of string of class names
    # Store new_dict as a text file in the save_dir
    dict_as_text = str(new_dict)
    dict_name = 'your_dictionary_with' + str(len(classes)) + 'classes.txt'
    dict_path = os.path.join(save_dir, dict_name)
    with open(dict_path, 'w') as x_file:
        x_file.write(dict_as_text)


    # Print dataset lengths
    print('Number of batches in ds_train:', tf.data.experimental.cardinality(ds_train).numpy())
    print('Number of batches in ds_valid:', tf.data.experimental.cardinality(ds_valid).numpy())

    # Use the datasets
    for image_batch, label_batch in ds_train.take(1):
        print("Image batch shape:", image_batch.shape)
        print("Label batch shape:", label_batch.shape)

    # Extract a batch of images from the training dataset
    for image_batch, label_batch in ds_train.take(1):
        # Convert to numpy array for inspection
        image_array = image_batch.numpy()
        # Print shape of the batch
        print("Batch shape:", image_array.shape)
        # Print min and max values in the batch to check the scale
        print("Min pixel value:", np.min(image_array))
        print("Max pixel value:", np.max(image_array))


    # Unique classes (labels) as a list
    print(f'Unique classes (labels) as a list: {class_names}')

    ############################################################################

    # Définition des répertoires
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, 'Modeles_keras')
    # Chemin des poids locaux
    weights_path = os.path.join(model_dir, f"{yourmodel_name}.h5")

    # inputs = layers.Input(shape=img_shape)
    base_model = eval(
        f'keras.applications.{yourmodel_name}(include_top=False, weights=None,input_shape=img_shape)')
    base_model.load_weights(weights_path)
    base_model.trainable = yourfreeze

    inputs = keras.Input(shape=img_shape)

    #x = img_augmentation(inputs)


    x = layers.Rescaling(scale=1. / 127.5, offset=-1)(inputs)
    # x = layers.Rescaling(scale=1./255)(x)

    x = base_model(x)

    if Useflattenlayer:
        x = keras.layers.Flatten()(x)
        print("Use a flatten layer to connect tridimentional layers to an unidimentional layer")
        if YourDropout > 0:
            x = keras.layers.Dropout(YourDropout, seed=123)(x)  # Regularize with dropout
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
        print("Use GlobalAveragePooling2D to connect tridimentional layers to an unidimentional layer")

    # x = keras.layers.GlobalAveragePooling2D()(x)  # GlobalMaxPooling2D

    # x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    # x= keras.layers.Dense(yoursizeofadditionallayer, kernel_regularizer=regularizers.L1L2(l1=0.016, l2=0.016)
    # , bias_regularizer=regularizers.L2(0.006), activity_regularizer=regularizers.L2(0.006), activation='relu')(x)
    if yoursizeofadditionallayer1 > 0:
        x = keras.layers.Dense(yoursizeofadditionallayer1, activation='relu')(x)
        if YourDropout > 0:
            x = keras.layers.Dropout(YourDropout, seed=123)(x)  # Regularize with dropout
    if yoursizeofadditionallayer2 > 0:
        x = keras.layers.Dense(yoursizeofadditionallayer2, activation='relu')(x)
        if YourDropout > 0:
            x = keras.layers.Dropout(YourDropout, seed=123)(x)  # Regularize with dropout
    if yoursizeofadditionallayer3 > 0:
        x = keras.layers.Dense(yoursizeofadditionallayer3, activation='relu')(x)
        if YourDropout > 0:
            x = keras.layers.Dropout(YourDropout, seed=123)(x)  # Regularize with dropout



    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax', name="layerfinal")(x)
    #outputs = keras.layers.Dense(NUM_CLASSES, activation='adamax', name="layerfinal")(x)
    model = Model(inputs, outputs)

    # model.summary(show_trainable=True)
    model.summary(show_trainable=True)



    my_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=yourstop_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=yourfactor, patience=yourpatience),
    ]

    model.compile(optimizer=keras.optimizers.Adamax(learning_rate=yourlearningrate),
                  loss=keras.losses.CategoricalCrossentropy(name="categorical_crossentropy"),
                  metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)],
    )

    history = model.fit(ds_train, epochs=yournumberofepochs, validation_data=ds_valid, callbacks=my_callbacks, verbose=1)

    # Récupérer la dernière valeur de val_categorical_accuracy
    last_val_accuracy = history.history['val_categorical_accuracy'][-1]
    print(f"La dernière valeur de val_categorical_accuracy est : {last_val_accuracy}")
    acc = last_val_accuracy * 100

    subject = yoursubject

    save_id = str(yourmodel_name + '-' + subject + '-resolution' + str(yourheight) + 'x' + str(yourwidth) + '_-_' + str(acc)[:str(acc).rfind('.') + 3] + 'percent_accuracy.keras')
    save_loc = os.path.join(save_dir, save_id)
    keras.saving.save_model(model, save_loc)
    print("The model has been trained and saved. You can now proceed to the next page => Evaluate performance")

#############################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa###################################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
def visualize(input_height, input_width, model_path, test_folder):
    save_dir = test_folder
    img_size = (input_height, input_width)
    # Répertoire des images
    base_dir = os.path.dirname(__file__)
    images_dir = os.path.join(base_dir, 'images_interface')

    test_dataset = keras.utils.image_dataset_from_directory(
        test_folder,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=32,
        image_size=img_size,
        shuffle=False,
        seed=123,
        validation_split=None,
        subset=None,
    )

    model = keras.saving.load_model(model_path, custom_objects=None, compile=True, safe_mode=True)


    def print_in_color(txt_msg,fore_tupple,back_tupple,):
        #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
        #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
        rf,gf,bf=fore_tupple
        rb,gb,bb=back_tupple
        msg='{0}' + txt_msg
        mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
        print(msg .format(mat), flush=True)
        print('\33[0m', flush=True) # returns default print color to back to black
        return

    def print_info(test_dataset, preds, print_code, save_dir):
        class_names = test_dataset.class_names  # Get class names
        new_dict = {i: class_name for i, class_name in
                    enumerate(class_names)}  # Create a dictionary {integer: class_name}
        classes = list(new_dict.values())  # List of string of class names

        # Store new_dict as a text file in the save_dir
        dict_as_text = str(new_dict)
        dict_name = 'your_dictionary_with' + str(len(classes)) + 'classes.txt'
        dict_path = os.path.join(save_dir, dict_name)
        with open(dict_path, 'w') as x_file:
            x_file.write(dict_as_text)

        y_true = []
        y_pred = []
        file_names = []


        # Collect true labels and image paths from the dataset
        for image_batch, label_batch in test_dataset:
            y_true.extend(label_batch.numpy())  # Collect true labels as they are (categorical)
            file_names.extend([f"Image {i}" for i in range(len(label_batch))])  # Placeholder for filenames

        # Convert one-hot encoded labels to class indices
        y_true_indices = np.argmax(y_true, axis=1)

        errors = 0
        error_list = []
        true_class = []
        pred_class = []
        prob_list = []
        error_indices = []

        for i, p in enumerate(preds):
            pred_index = np.argmax(p)
            true_index = y_true_indices[i]
            y_pred.append(pred_index)

            if pred_index != true_index:  # A misclassification has occurred
                error_list.append(file_names[i])  # Placeholder for actual file paths
                true_class.append(new_dict[true_index])
                pred_class.append(new_dict[pred_index])
                prob_list.append(p[pred_index])
                error_indices.append(true_index)
                errors += 1

        if print_code != 0:
            if errors > 0:
                r = min(print_code, errors)
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class', 'True Class',
                                                                'Probability')
                print_in_color(msg, (0, 255, 0), (55, 65, 80))
                for i in range(r):
                    msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(
                        error_list[i], pred_class[i], true_class[i], ' ', prob_list[i])
                    print_in_color(msg, (255, 255, 255), (55, 65, 60))
            else:
                msg = 'With accuracy of 100% there are no errors to print'
                print_in_color(msg, (0, 255, 0), (55, 65, 80))

        if errors > 0:
            plot_bar = []
            plot_class = []
            for key, value in new_dict.items():
                count = error_indices.count(key)
                if count != 0:
                    plot_bar.append(count)  # List containing how many times a class had an error
                    plot_class.append(value)  # Stores the class
            fig = plt.figure()
            fig.set_figheight(len(plot_class) / 3)
            fig.set_figwidth(10)
            plt.style.use('fivethirtyeight')
            for i in range(len(plot_class)):
                c = plot_class[i]
                x = plot_bar[i]
                plt.barh(c, x)
                plt.title('Errors by Class on Test Set')
            plt.savefig(os.path.join(save_dir, 'errors_by_class.png'),
                        bbox_inches='tight')  # Ligne à insérer avant plt.show()
            plt.savefig(os.path.join(images_dir, "errors_by_class.png"), bbox_inches='tight')
            # plt.show()
        else:
            # Chemins source et destination
            source = os.path.join(images_dir, "no_error.png")
            destination = os.path.join(images_dir, "errors_by_class.png")
            # Copier le fichier et renommer
            shutil.copyfile(source, destination)
            destination = os.path.join(save_dir, "errors_by_class.png")
            # Copier le fichier et renommer
            shutil.copyfile(source, destination)

        y_pred = np.array(y_pred)

        # Handle missing
        # classes in predictions or true labels
        true_labels_available = set(y_true_indices)
        predicted_labels_available = set(y_pred)

        # If there are missing labels in either y_true or y_pred, we need to handle that
        all_classes = set(range(len(classes)))
        missing_true_classes = all_classes - true_labels_available
        missing_pred_classes = all_classes - predicted_labels_available

        # Create padded confusion matrix
        if len(classes) <= 150:
            cm = confusion_matrix(y_true_indices, y_pred, labels=list(all_classes))  # Include all possible classes
            length = len(classes)
            fig_width = int(length * 0.5)
            fig_height = int(length * 0.5)
            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
            plt.xticks(np.arange(length) + .5, classes, rotation=90)
            plt.yticks(np.arange(length) + .5, classes, rotation=0)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'),
                        bbox_inches='tight')  # Ligne à insérer avant plt.show()
            plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), bbox_inches='tight')
            #plt.tight_layout()  # Ajoute cette ligne pour ajuster les marges avant plt.show()
            # plt.show()

        ##################aa
        # Compter les occurrences pour chaque classe
        class_counts = np.bincount(y_true_indices, minlength=len(classes))

        # Vérifier si chaque classe a au moins une occurrence
        missing_classes = [classes[i] for i, count in enumerate(class_counts) if count == 0]


        # Calculate AUC-ROC scores
        y_true_one_hot = np.array(y_true)
        auc_scores = {}
        for i, class_name in enumerate(classes):
            try:
                auc = roc_auc_score(y_true_one_hot[:, i], preds[:, i])
                auc_scores[class_name] = auc
            except ValueError:
                auc_scores[class_name] = None  # No positive samples for this class

        if len(set(y_true_indices)) > 1:  # Vérifie qu'il y a plus d'une classe
            if missing_classes:
                print(f"Warning: The following classes have no samples in y_true: {missing_classes}")
                overall_auc = 555555  # Pas de calcul possible
            else:
                overall_auc = roc_auc_score(y_true_one_hot, preds, average="macro")
        else:
            overall_auc = None  # Pas de calcul possible
            print("Warning: ROC AUC score cannot be calculated because only one class is present in y_true.")
        ###################aa

        # Handle the classification report with missing classes
        clr = classification_report(
            y_true_indices,
            y_pred,
            target_names=classes,
            labels=list(all_classes),
            zero_division=0  # This avoids division by zero for classes with no true samples
        )
        print("Classification Report:\n----------------------\n", clr)

        # Handle the classification report with missing classes
        clr2 = classification_report(
            y_true_indices,
            y_pred,
            target_names=classes,
            labels=list(all_classes),
            zero_division=0,  # This avoids division by zero for classes with no true samples
            output_dict=True  # For easier manipulation
        )
        print("Classification Report:\n----------------------\n", clr2)

        ######################aaa
        # Add AUC-ROC to the classification report
        for class_name, auc in auc_scores.items():
            if class_name in clr2:
                clr2[class_name]["AUC-ROC"] = auc if auc is not None else "N/A"

        # Convert the classification report back to string format
        clr_text = "Classification Report:\n----------------------\n"
        clr_text += f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<12}{'AUC-ROC':<12}\n"
        clr_text += "-" * 80 + "\n"

        for class_name in classes:
            metrics = clr2[class_name]
            auc_score = metrics.get("AUC-ROC", "N/A")
            clr_text += f"{class_name:<20}{metrics['precision']:<12.2f}{metrics['recall']:<12.2f}{metrics['f1-score']:<12.2f}{metrics['support']:<12.0f}{auc_score:<12}\n"

        # Add accuracy, macro avg, and weighted avg
        for key in ["accuracy", "macro avg", "weighted avg"]:
            if key in clr2:
                if key == "accuracy":
                    clr_text += f"\n{'Accuracy':<20}{'':<12}{'':<12}{clr2[key]:<12.2f}{'':<12}{'':<12}\n"
                else:
                    avg_metrics = clr2[key]
                    clr_text += f"{key.title():<20}{avg_metrics['precision']:<12.2f}{avg_metrics['recall']:<12.2f}{avg_metrics['f1-score']:<12.2f}{avg_metrics['support']:<12.0f}\n"

        # Add global AUC-ROC
        if overall_auc is not None:
            if missing_classes:
                clr_text += "\nWarning: Overall AUC-ROC not defined (you must have at least one image by classe)\n"
            else:
                clr_text += "\nOverall AUC-ROC: {:.4f}\n".format(overall_auc)
        else:
            clr_text += "\nWarning: Overall AUC-ROC not defined (only one class present)\n"

        # Save the classification report
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(clr_text)
        report_path = os.path.join(images_dir, 'classification_report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(clr_text)


        print(clr_text)
        ######################aaa

        """report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write("Classification Report:\n----------------------\n")
            report_file.write(clr)
        report_path = os.path.join(images_dir, 'classification_report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write("Classification Report:\n----------------------\n")
            report_file.write(clr)"""

    print_code = 0
    preds =  model.predict(test_dataset)
    print(preds)
    print_info(test_dataset, preds, print_code, save_dir)
    print("Performance has been evaluated. If the results match your expectations, you can use your model for sorting. Before annotating and sorting new images, you need to remove the metadata (EXIF) from your photos on the next page => Remove metadata")

#############################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa###################################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

def Annotate(model_path, dictionnary_path, input_dir, output_dir):
    # Load the dictionary from the text file
    print("Annotation and sorting in progress…")
    dict_file_path = dictionnary_path
    with open(dict_file_path, 'r') as file:
        class_dict = ast.literal_eval(file.read())

    # Load the model
    m = keras.models.load_model(model_path)
    dirnamea = output_dir  # Destination directory
    dirnameb = input_dir  # Source directory

    # Create directories based on the class names in the dictionary
    for class_name in class_dict.values():
        newfq = os.path.join(dirnamea, class_name)
        os.makedirs(newfq, exist_ok=True)

    # Create mappings and extract the first 3 letters of each class name
    reverse_mapping = class_dict
    class_abbr = {k: v[:3] for k, v in reverse_mapping.items()}  # Create abbreviations from class names

    def mapper(value):
        return reverse_mapping[value]

    # List of supported image extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    def get_target_size_from_filename(model_path):
        # Use a regular expression to find the resolution in the filename (e.g., "resolution224x224")
        match = re.search(r'resolution(\d+)x(\d+)_-_', model_path)

        if match:
            # Extract the width and height
            width, height = map(int, match.groups())
            return (width, height)
        else:
            raise ValueError("Resolution not found in the model file name")

    target_size = get_target_size_from_filename(model_path)


    # Process and classify images
    for dirname, _, filenames in os.walk(dirnameb):
        print(dirname)
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):  # Check for supported extensions
                originalpict = os.path.join(dirname, filename)

                # Preprocess the image for the model
                image = keras.utils.load_img(originalpict, target_size=target_size)
                image = keras.utils.img_to_array(image)
                image = np.expand_dims(image, axis=0)

                # Get the prediction and probabilities
                prediction = m(image, training=False)
                prediction2 = np.round(prediction * 100).astype(int)  # Convert to percentages

                # Print probabilities with class abbreviations
                prob_output = [f"{class_abbr[i]}: {prob}%" for i, prob in enumerate(prediction2[0])]
                #print("Probabilities: ", prob_output)

                # Determine the class with the highest probability
                value = np.argmax(prediction)
                move_name = mapper(value)

                # Move and rename the file
                newf = os.path.join(dirnamea, move_name)
                shutil.copy(originalpict, newf)
                filecopied = os.path.join(newf, filename)

                # Format prediction for renaming
                prob_str = "_".join([f"{class_abbr[i]}{prob}%" for i, prob in enumerate(prediction2[0])])
                filerenamed = os.path.join(newf, f"{prob_str}_{filename}")

                # Rename the copied file with the formatted predictions
                os.rename(filecopied, filerenamed)
    print("Annotation and sorting completed. You can view the results by clicking on the 6th button.")

#############################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa###################################aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

###################################################################################################################################################

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

export_folder_path = ""
import_folder_path = ""



class Window(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisuelAIclassification")
        self.geometry('1280x800')
        self.configure(fg_color='gray78')

        # Create pages
        self.create_page_EXIF() # Ensure all frames are initialized before use
        self.create_page_Augment()
        self.create_page_generate_model()
        self.create_page_Visualize()
        self.create_use_model()
        #self.create_Annotate()
        self.home_page()  # Call the home page after all pages are created

        self.ribbon_frame = ctk.CTkFrame(self, corner_radius=0, fg_color='gray60')
        self.ribbon_frame.place(relx=0, rely=0, relwidth=1, relheight=0.05, anchor="nw")

        # Create buttons on the ribbon
        self.ribbon_buttons = {}
        self.create_ribbon_buttons()

        # Set default page
        self.show_frame(self.home_frame)

    def create_ribbon_buttons(self):
        self.ribbon_buttons = {}
        self.active_button = None  # Track the currently active button

        button_texts = [
            ("Home", self.home_frame),
            ("Prepare your dataset", self.page_Augment),
            ("Train your model", self.create_model_main_page),
            ("Evaluate performance", self.page_Visualize),
            ("Remove metadata", self.page_EXIF),
            ("Annotate and sort", self.page_UseModel)
        ]

        for text, frame in button_texts:
            button = ctk.CTkButton(
                self.ribbon_frame,
                text=text,
                command=lambda f=frame, b=text: self.show_frame(f, b),
                fg_color="#2980b9",
                text_color="white",
                hover_color="#3498db",
                corner_radius=10
            )
            button.pack(side="left", padx=5, pady=5)
            self.ribbon_buttons[text] = button  # Store the button with its text key

    def home_page(self):
        # Main frame with consistent color to match the rest of the UI
        self.home_frame = ctk.CTkFrame(self, fg_color="#2c3e50", corner_radius=0)
        self.home_frame.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Create a subframe to act as a padded container for buttons
        self.button_panel = ctk.CTkFrame(self.home_frame, fg_color="#34495e", corner_radius=20)
        self.button_panel.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.7)

        # List of buttons with matching styles
        buttons = [
            ("1 Prepare your data for Image Recognition Models", self.page_Augment),
            ("2 Train your model", self.create_model_main_page),
            ("3 Evaluate the performance of your model (on the test folder)", self.page_Visualize),
            ("4 Remove metadata (EXIF) from your pictures", self.page_EXIF),
            ("5 Use your model for annotation and sorting", self.page_UseModel)
        ]

        # Spacing adjustments
        button_spacing = 0.02  # Adjust the space between buttons
        start_relx = 0.1

        # Creating buttons with updated styling
        for i, (text, command) in enumerate(buttons):
            # Wrap text for the button to fit its size
            wrapped_text = self.wrap_text(text, 20)  # Adjust the second parameter (max_length) as needed

            button = ctk.CTkButton(
                self.button_panel,
                text=wrapped_text,
                command=lambda f=command: self.show_frame(f),
                fg_color="#2980b9",  # Consistent blue color
                text_color="white",
                hover_color="#3498db",  # Subtle blue hover effect
                corner_radius=15,  # Rounded corners
                font=("Arial", 14, "bold"),  # Bold and slightly larger font
                border_spacing=0
            )

            # Calculate dynamic button position and size
            rel_width = 0.18  # Set a consistent width for each button
            rel_x = start_relx + (rel_width + button_spacing) * i  # Calculate x position
            rel_y = 0.5  # Center the buttons vertically

            button.place(relx=rel_x, rely=rel_y, anchor="center", relwidth=rel_width, relheight=0.6)

    def wrap_text(self, text, max_length):
        """
        Wrap text to fit within the max_length, ensuring the first character
        (e.g., numbering) is placed on its own line.
        """
        # Split the text into the first character and the rest
        first_char, remaining_text = text[0], text[1:].strip()

        # Start with the first character on its own line
        wrapped_lines = first_char + "\n"

        # Process the remaining text
        words = remaining_text.split()
        current_line = ""

        for word in words:
            # Check if adding the next word exceeds the max_length
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                # If it exceeds, add the current line to the wrapped text
                wrapped_lines += current_line + "\n"
                current_line = word  # Start a new line with the current word

        # Add any remaining text
        wrapped_lines += current_line
        return wrapped_lines.strip()  # Remove any trailing newline

    def create_folder_import_button(self, page_name, folder_name, title, container, rel_x, rel_y):
        # Create the full name by combining page name and folder name
        full_name = f"{page_name}_{folder_name}"

        # Initialize the path to "Select a folder" for the first display
        self.folder_paths[full_name] = "Select a folder"

        # Create the import button
        button = ctk.CTkButton(
            container,
            text=title,
            command=lambda: self.select_folder(full_name, title),
            fg_color="#2980b9",
            text_color="white",
            hover_color="#3498db",
            corner_radius=10
        )
        button.place(relx=rel_x, rely=rel_y, anchor="center")

        # Create the label to show the selected folder path
        label = ctk.CTkLabel(
            container,
            text=self.folder_paths[full_name],
            text_color="white",
            font=("Arial", 12)
        )
        label.place(relx=rel_x, rely=rel_y + 0.05, anchor="center")

        # Store the label widget in a dictionary for later updates
        setattr(self, f"{full_name}_label", label)

    def select_folder(self, full_name, title):
        # Open a file dialog to select a folder
        selected_folder = tk.filedialog.askdirectory(title=title)

        if selected_folder:
            # Update the folder path dictionary
            self.folder_paths[full_name] = selected_folder
            print(f"Selected folder path for {full_name}: {selected_folder}")

            # Update the corresponding label text to show the selected folder
            label = getattr(self, f"{full_name}_label", None)
            if label:
                label.configure(text=selected_folder)

    def create_option_menu(self, page_name, menu_name, options, title, container, rel_x, rel_y):
        # Create the full name by combining page name and menu name
        full_name = f"{page_name}_{menu_name}"

        # Initialize the selected option with the first option or a default
        self.selected_options[full_name] = tk.StringVar(value=options[0])

        # Create the label to display the title
        label = ctk.CTkLabel(container, text=title, text_color="white", font=("Arial", 12))
        label.place(relx=rel_x, rely=rel_y, anchor="nw")

        # Create the option menu
        #def optionmenu_callback(choice):
        #    print("sssssssssssssssssssssss", choice)


        option_menu = ctk.CTkOptionMenu(
            container,
            variable=self.selected_options[full_name],
            values=options,
            fg_color="#2980b9",
            text_color="white",
            button_color="#3498db",
            button_hover_color="#2980b9",
            command=lambda choice: self.update_selected_option(full_name, choice)
        )
        option_menu.place(relx=rel_x + 0.15, rely=rel_y, anchor="nw")

    def update_selected_option(self, full_name, choice):
        # Met à jour l'option sélectionnée dans le dictionnaire
        self.selected_options[full_name].set(choice)
        print(f"Updated selected option for {full_name}: {self.selected_options[full_name].get()}")

    def get_selected_option(self, page_name, menu_name):
        # Get the full name to fetch the selected option
        full_name = f"{page_name}_{menu_name}"
        return self.selected_options[full_name].get()



    def create_page_EXIF(self):
        # Initial folder paths for import and export
        self.import_folder_pathEXIF = "Select a folder"
        self.export_folder_pathEXIF = "Select a folder"

        # Main EXIF page frame with a soft background color
        self.page_EXIF = Frame(self, color="#2c3e50", corner_radius=0)
        self.page_EXIF.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Left panel for folder selection and buttons
        self.left_panel_EXIF = Frame(self.page_EXIF, color="#34495e", corner_radius=0)
        self.left_panel_EXIF.place(relx=0, rely=0, relwidth=0.5, relheight=1, anchor="nw")

        # Right panel for explanation text with a contrasting color
        self.right_panel = Frame(self.page_EXIF, color="#2c3e50")
        self.right_panel.place(relx=0.5, rely=0, relwidth=0.5, relheight=1, anchor="nw")

        # Explanation text box with rounded corners and improved readability
        self.explication = ctk.CTkTextbox(
            self.right_panel,
            fg_color="#7f8c8d",
            corner_radius=20,
            font=("Arial", 14, "italic"),
            text_color="white",
            wrap="word"
        )

        # Insert explanatory content
        text_content = """Metadata in digital images, including EXIF data, often contains non-essential information such as the date, time, and location of capture. While not required for training models, this metadata can increase dataset size and potentially introduce issues during preprocessing.

Benefits of Metadata Removal
• Dataset Size Reduction: Removing metadata decreases the overall size of the dataset, making it more manageable.
• Enhanced Anonymization: Eliminating details such as timestamps and geolocation ensures better privacy and compliance with data protection standards.
• Improved Data Integrity: By stripping metadata, the process ensures the images are ready for machine learning workflows without unintended misvectorization or encoding issues (e.g.,error "XXX extraneous bytes before marker 0xd9")


Step-by-step guide:

    1) Select the folder containing your images. This folder can house subfolders organized by categories or directly contain the images.

    2) Specify the output folder. The processed images, stripped of metadata, will be saved here.

    3) Press the "Remove Metadata" button to initiate the conversion process.

Outputs
    Once the metadata removal is complete, the images will be saved in the output folder, ready for use.
    
By automating metadata removal, this module ensures that datasets are streamlined, anonymized, and optimized for image classification tasks.
"""
        self.explication.insert("1.0", text_content)
        self.explication.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9,
                               relheight=0.9)  # Centered in the right panel

        # Button to select import folder
        self.import_dataEXIF = ctk.CTkButton(
            self.left_panel_EXIF,
            text="1) Select Import Folder",
            command=self.open_import_folderEXIF,
            fg_color="#2980b9",  # Blue color for a noticeable call to action
            text_color="white",
            hover_color="#3498db",  # Hover effect to provide feedback
            corner_radius=10
        )
        self.import_dataEXIF.place(relx=0.5, rely=0.2, anchor="center")

        # Label to show selected import folder path
        self.import_folder_labelEXIF = ctk.CTkLabel(
            self.left_panel_EXIF,
            text=self.import_folder_pathEXIF,
            text_color="white",
            font=("Arial", 12)
        )
        self.import_folder_labelEXIF.place(relx=0.5, rely=0.25, anchor="center")

        # Button to select export folder
        self.export_dataEXIF = ctk.CTkButton(
            self.left_panel_EXIF,
            text="2) Select Export Folder",
            command=self.open_export_folderEXIF,
            fg_color="#2980b9",
            text_color="white",
            hover_color="#3498db",
            corner_radius=10
        )
        self.export_dataEXIF.place(relx=0.5, rely=0.35, anchor="center")

        # Label to show selected export folder path
        self.export_folder_labelEXIF = ctk.CTkLabel(
            self.left_panel_EXIF,
            text=self.export_folder_pathEXIF,
            text_color="white",
            font=("Arial", 12)
        )
        self.export_folder_labelEXIF.place(relx=0.5, rely=0.4, anchor="center")

        # Button to trigger EXIF removal
        self.remove_exif_button = ctk.CTkButton(
            self.left_panel_EXIF,
            text="3) Remove Metadata",
            command=self.remove_exif_action,
            fg_color="#e74c3c",  # Red to signify action
            text_color="white",
            hover_color="#c0392b",  # Darker red on hover
            corner_radius=10,
            font=("Arial", 13, "bold")
        )
        self.remove_exif_button.place(relx=0.5, rely=0.6, anchor="center")

        # Ajouter un encadré pour afficher le contenu du terminal
        self.terminal_textbox = ctk.CTkTextbox(self.left_panel_EXIF, wrap="word")
        self.terminal_textbox.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.99, relheight=0.15)
        # Optionnel : définir du texte par défaut
        self.terminal_textbox.insert("1.0", "Code output...")
        # Initialize stdout redirector without the callback since we don't need display_images
        self.stdout_redirector = StdoutRedirector(
            callback=None,  # Pas besoin de callback pour juste afficher le terminal
            callback_args=None,
            complete_message="",  # Pas de message complet nécessaire ici
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )

        # Commencez la redirection du stdout vers le terminal_textbox
        self.stdout_redirector.start()

    def open_import_folderEXIF(self):
        # Open a file dialog to select the import folder
        selected_folder = tk.filedialog.askdirectory(title="Select Import Folder for EXIF")

        if selected_folder:
            # Update the import folder path variable for EXIF page
            self.import_folder_pathEXIF = selected_folder
            print(f"Selected import folder path for EXIF: {self.import_folder_pathEXIF}")

            # Update the label text to show the selected folder
            self.import_folder_labelEXIF.configure(text=self.import_folder_pathEXIF)

    def open_export_folderEXIF(self):
        # Open a file dialog to select the export folder
        selected_folder_exportEXIF = tk.filedialog.askdirectory(title="Select Export Folder for EXIF")

        if selected_folder_exportEXIF:
            # Update the export folder path variable for EXIF page
            self.export_folder_pathEXIF = selected_folder_exportEXIF
            print(f"Selected export folder path for EXIF: {self.export_folder_pathEXIF}")

            # Update the label text to show the selected folder
            self.export_folder_labelEXIF.configure(text=self.export_folder_pathEXIF)

    def remove_exif_action(self):
        # Check if import and export folder paths are set
        if not self.import_folder_pathEXIF or not self.export_folder_pathEXIF:
            # Show a warning message in the GUI or log the error
            print("Please select valid import and export folders.")
            return

        try:
            # Call the remove EXIF function with the selected paths
            remove_exif_from_jpg(self.import_folder_pathEXIF, self.export_folder_pathEXIF)
            #print("EXIF data removal completed successfully.")
            # Optionally, show a confirmation message in the GUI
        except Exception as e:
            print(f"Error during EXIF removal: {e}")
            # Optionally, display an error message in the GUI

    def create_page_Augment(self):
        # Initial folder paths for import and export
        self.import_folder_pathAUGMENT = "Select a folder"
        self.export_folder_pathAUGMENT = "Select a folder"
        self.fill_mode = tk.StringVar(value ="constant")

        # Main augmentation page frame with a soft background color
        self.page_Augment = Frame(self, color="#2c3e50", corner_radius=0)
        self.page_Augment.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Left panel for folder selection
        self.left_panel_Augment = Frame(self.page_Augment, color="#34495e", corner_radius=0)
        self.left_panel_Augment.place(relx=0, rely=0, relwidth=0.6, relheight=1, anchor="nw")

        # Right panel for explanation text
        self.right_panel_Augment = Frame(self.page_Augment, color="#2c3e50")
        self.right_panel_Augment.place(relx=0.6, rely=0, relwidth=0.4, relheight=1, anchor="nw")

        # Box for explanations
        self.explication_Augment = ctk.CTkTextbox(
            self.right_panel_Augment,
            fg_color="#7f8c8d",
            corner_radius=20,
            font=("Arial", 14, "italic"),
            text_color="white",
            wrap="word"
        )

        text_content_Augment = """The dataset creation module is designed to streamline the preparation of training, validation, and test datasets required for training an image recognition model. This module incorporates data augmentation, a technique used to artificially expand datasets and improve model performance by introducing variability while preserving the essential features of the images.

Overview of Data Augmentation:
    -Data augmentation applies random transformations such as rotation (0 to 0.05 radians), translation (up to 10% horizontally and vertically), and flipping (horizontal and vertical) to existing images. These transformations enhance the dataset by making the model more robust to spatial variations, ensuring that an object in the image remains identifiable in its initial category (e.g., a dog remains a dog).
    -This process is particularly beneficial for balancing the dataset when one or more categories are underrepresented. By increasing the number of images in these underrepresented classes to match the most represented class, data augmentation mitigates bias and improves the model's ability to generalize across all categories.
    -Additionally, this module automatically divides your dataset into training (80%), validation (10%), and test (10%) datasets, crucial for model training and evaluation.

Input Requirements:
    -Images must be in .jpg, .jpeg, .png, or .bmp format.
    -The dataset should be organized into subfolders for each category.
    -EXIF metadata is removed during this process to ensure compatibility and prevent encoding errors.
    
    
Step-by-step guide:

    1) Select the folder containing your images. This folder should either contain subfolders for each category or directly house the images.

    2)  Choose an output folder. The module will create three subfolders within this directory: 
        -Training Folder: 80% of the images.
        -Validation Folder: 10% of the images.
        -Test Folder: 10% of the images.

    3) Choose the data augmentation filling type. When data augmentation is applied, random rotation, flipping, and translation will shift the image. As a result, you will need to fill the borders, and you can choose the filling technique to use. "Constant" fills the outer borders with black, "Nearest" extends from the nearest pixel, "Reflect" creates a reflection, and "Wrap" reintroduces the cut-off parts to fill the gaps.

    4) Decide where to apply augmentation:
    -Training dataset: Essential for teaching the model by exposing it to various data patterns.
    -Validation dataset: Used during training to evaluate and fine-tune the model, ensuring it generalizes well to unseen data.
    -Test dataset: Reserved for final performance evaluation on unseen data.
   
    Choose if you want to apply this augmentation to the training, validation, and/or test folders. Some people avoid applying augmentation to the validation or test datasets, as doing so can artificially inflate the model's performance by introducing data variations it has already seen during training.
    Note: Applying augmentation to the validation or test datasets may artificially inflate performance metrics since variations introduced during augmentation may already be familiar to the model. If you wish to create datasets without augmentation, leave all checkboxes in steps 3 and 4 unchecked.

    5) Start the process.

Outputs and Verification
    Once the process is complete, you can verify the results by checking the subfolders in the output directory. The augmented images will be organized according to the specified categories, and their quality and interpretability can be assessed directly on your computer.
    By ensuring systematic dataset preparation, this module facilitates efficient and effective model training while enhancing dataset robustness through augmentation."""
        self.explication_Augment.insert("1.0", text_content_Augment)
        self.explication_Augment.place(relx=0.5, rely=0.42, anchor="center", relwidth=0.95,
                                       relheight=0.8)  # Centered in the right panel

        # Button to select import folder
        self.import_dataAUGMENT = ctk.CTkButton(
            self.left_panel_Augment,
            text="1) Select Import Folder",
            command=self.open_import_folderAUGMENT,
            fg_color="#2980b9",
            text_color="white",
            hover_color="#3498db",
            corner_radius=10
        )
        self.import_dataAUGMENT.place(relx=0.5, rely=0.10, anchor="center")

        # Label to show selected import folder path
        self.import_folder_labelAUGMENT = ctk.CTkLabel(
            self.left_panel_Augment,
            text=self.import_folder_pathAUGMENT,
            text_color="white",
            font=("Arial", 12)
        )
        self.import_folder_labelAUGMENT.place(relx=0.5, rely=0.15, anchor="center")

        # Button to select export folder
        self.export_dataAUGMENT = ctk.CTkButton(
            self.left_panel_Augment,
            text="2) Select Export Folder",
            command=self.open_export_folderAUGMENT,
            fg_color="#2980b9",
            text_color="white",
            hover_color="#3498db",
            corner_radius=10
        )
        self.export_dataAUGMENT.place(relx=0.5, rely=0.20, anchor="center")

        # Label to show selected export folder path
        self.export_folder_labelAUGMENT = ctk.CTkLabel(
            self.left_panel_Augment,
            text=self.export_folder_pathAUGMENT,
            text_color="white",
            font=("Arial", 12)
        )
        self.export_folder_labelAUGMENT.place(relx=0.5, rely=0.25, anchor="center")

        # Image labels for augmentation modes
        image_paths = {
            "constant": "images_interface/constant_img.jpg",
            "nearest": "images_interface/nearest_img.jpg",
            "reflect": "images_interface/reflect_img.jpg",
            "wrap": "images_interface/wrap_img.jpg"
        }


        img = Image.open(resource_path("images_interface/initial_img.jpg")).resize((140, 140))
        img_tk = ImageTk.PhotoImage(img)
        label = ctk.CTkLabel(self.left_panel_Augment, image=img_tk, text="", bg_color='#34495e')
        label.image = img_tk
        label.place(relx=0.02, rely=0.32, anchor="nw")
        self.labelInitial = ctk.CTkLabel(self.left_panel_Augment, text="Initial Image",
                                   text_color="white")
        self.labelInitial.place(relx=0.05, rely=0.53, anchor="nw")



        # Fill mode variables for each checkbox
        self.fill_mode_vars = {
            "constant": tk.BooleanVar(value=False),  # Default selected mode
            "nearest": tk.BooleanVar(value=False),
            "reflect": tk.BooleanVar(value=False),
            "wrap": tk.BooleanVar(value=False),
        }

        def update_fill_mode(selected_mode):
            """
            Update fill mode and ensure mutual exclusivity.
            If the same mode is clicked again, all modes are deselected.
            """
            # Check if the current mode is already selected
            if self.fill_mode.get() == selected_mode:
                # Deselect all modes
                self.fill_mode.set("")
                for var in self.fill_mode_vars.values():
                    var.set(False)
            else:
                # Update the selected fill mode and ensure mutual exclusivity
                self.fill_mode.set(selected_mode)
                for mode, var in self.fill_mode_vars.items():
                    var.set(mode == selected_mode)

########################

        # Frame for project name
        self.frame1 = ctk.CTkFrame(self.left_panel_Augment, fg_color='transparent', border_color='black', border_width=2)
        self.frame1.place(relx=0.20, rely=0.27, relwidth=0.8, relheight=0.40, anchor="nw")
        self.label1 = ctk.CTkLabel(self.frame1, text="3) Select the data augmentation fill mode type you want to use:", text_color="white")
        self.label1.place(relx=0.5, rely=0.08, anchor="center")

        # Place the images
        self.image_labels_Augment = {}
        for index, (mode, path) in enumerate(image_paths.items()):
            img = Image.open(path).resize((140, 140))
            img_tk = ImageTk.PhotoImage(img)
            label = ctk.CTkLabel(self.frame1, image=img_tk, text="", bg_color='#34495e')
            label.image = img_tk
            label.place(relx=0.01 + index * 0.25, rely=0.15, anchor="nw")
            self.image_labels_Augment[mode] = label

        for index, (mode, var) in enumerate(self.fill_mode_vars.items()):
            button = BoutonCocher(master=self.frame1,
                                  label_text=mode.capitalize(),
                                  variable=var,
                                  fg_color="#7f8c8d",
                                  hover_color="#95a5a6",
                                  border_color="#B0C4DE",
                                  command=lambda m=mode: update_fill_mode(m))
            button.place(relx=0.01 + index * 0.25, rely=0.65, anchor="nw")





        # Label Data Augmentation
        self.data_augmentation_label_Augment = ctk.CTkLabel(self.left_panel_Augment,
                                                            text="4) Data augmentation on:",
                                                            text_color="white")
        self.data_augmentation_label_Augment.place(relx=0.02, rely=0.73, anchor="nw")

        # Variables to store checkbox state
        var_training = ctk.BooleanVar()
        var_validation = ctk.BooleanVar()
        var_test = ctk.BooleanVar()

        check_buttons = {
            "Training dataset": var_training,
            "Validation dataset": var_validation,
            "Test dataset": var_test
        }
        for index, (label, var) in enumerate(check_buttons.items()):
            button = BoutonCocher(master=self.left_panel_Augment,
                                  label_text=label,
                                  variable=var,
                                  fg_color="#7f8c8d",
                                  hover_color="#95a5a6",
                                  border_color="#B0C4DE")
            button.place(relx=0.25 + index * 0.2, rely=0.73, anchor="nw")

        # Create Set Button
        self.create_set_button_Augment = ctk.CTkButton(self.left_panel_Augment,
                                                  text="5) Create Dataset",
                                                  command=lambda: create_and_augment(
                                                      self.pass_variable(self.import_folder_pathAUGMENT),
                                                      self.pass_variable(self.export_folder_pathAUGMENT),
                                                      self.pass_variable(self.fill_mode.get()),
                                                      augment_training=var_training.get(),
                                                      augment_validation=var_validation.get(),
                                                      augment_test=var_test.get()
                                                  ),
                                                  fg_color="#e74c3c",  # Red for action
                                                  text_color="white",
                                                  hover_color="#c0392b",  # Darker red on hover
                                                  corner_radius=10
                                                  )
        self.create_set_button_Augment.place(relx=0.5, rely=0.9, anchor="center")

        # Ajouter un encadré pour afficher le contenu du terminal
        self.terminal_textbox = ctk.CTkTextbox(self.right_panel_Augment, wrap="word")
        self.terminal_textbox.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.99, relheight=0.15)
        # Optionnel : définir du texte par défaut
        self.terminal_textbox.insert("1.0", "Code output...")
        # Initialize stdout redirector without the callback since we don't need display_images
        self.stdout_redirector = StdoutRedirector(
            callback=None,  # Pas besoin de callback pour juste afficher le terminal
            callback_args=None,
            complete_message="",  # Pas de message complet nécessaire ici
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )

        # Commencez la redirection du stdout vers le terminal_textbox
        self.stdout_redirector.start()

    def open_import_folderAUGMENT(self):
        # Open a file dialog to select the import folder for Augment page
        selected_folder_importAUGMENT = tk.filedialog.askdirectory(title="Select Import Folder for Augmentation")

        if selected_folder_importAUGMENT:
            # Update the import folder path variable for Augment page
            self.import_folder_pathAUGMENT = selected_folder_importAUGMENT
            print(f"Selected import folder path for Augment: {self.import_folder_pathAUGMENT}")

            # Update the label text to show the selected folder
            self.import_folder_labelAUGMENT.configure(text=self.import_folder_pathAUGMENT)

    def open_export_folderAUGMENT(self):
        global global_export_folder_pathAUGMENT
        # Open a file dialog to select the export folder for Augment page
        selected_folder_importAUGMENT = tk.filedialog.askdirectory(title="Select Export Folder for Augmentation")

        if selected_folder_importAUGMENT:
            # Update the export folder path variable for Augment page
            self.export_folder_pathAUGMENT = selected_folder_importAUGMENT
            global_export_folder_pathAUGMENT = selected_folder_importAUGMENT # Pour ouvrir le dossier dans Visualize
            print(f"Selected export folder path for Augment: {self.export_folder_pathAUGMENT}")

            # Update the label text to show the selected folder
            self.export_folder_labelAUGMENT.configure(text=self.export_folder_pathAUGMENT)

    def create_labeled_entry(self, parent_frame, label_text, tooltip_text, relx, rely, is_float=False,
                             default_value=None):
        # Create the label with the provided text and position
        label = ctk.CTkLabel(parent_frame, text=label_text, text_color="white")
        label.place(relx=relx, rely=rely, anchor="nw")

        # Add the tooltip icon next to the label
        self.create_icon_with_tooltip(parent_frame, relx=relx + 0.54, rely=rely, tooltip_text=tooltip_text)

        # Create the entry box for the number input (use IntVar or DoubleVar based on is_float)
        if is_float:
            entry_var = tk.DoubleVar()  # Use DoubleVar for floating point numbers
        else:
            entry_var = tk.IntVar()  # Use IntVar for integers

        # Set the default value if provided
        if default_value is not None:
            entry_var.set(default_value)

        entry_box = ctk.CTkEntry(parent_frame, textvariable=entry_var)
        entry_box.place(relx=relx + 0.42, rely=rely, relwidth=0.1, anchor="nw")

        # Bind events to save the entry value when focus is lost or Enter is pressed
        entry_box.bind("<FocusOut>", lambda event: self.save_value(entry_var))
        entry_box.bind("<Return>", lambda event: self.save_value(entry_var))

        # Return the entry variable to allow value retrieval
        return entry_var

    def create_page_generate_model(self):

        self.training_folder_pathGENERATE = ""
        self.validate_folder_pathGENERATE = ""
        self.test_folder_pathGENERATE = ""
        self.output_folder_pathGENERATE = ""

        # Icon for hovering with the mouse and display message
        image_icon_path = resource_path("images_interface/point_dinterrogation.png")  # Provide the path to your image
        image_icon = Image.open(image_icon_path)
        image_icon = image_icon.resize((40, 40), Image.LANCZOS)  # Resize the image
        self.image_icon = ctk.CTkImage(light_image=image_icon, size=(18, 18))


        # Define IntVar for number of epochs
        self.yourNumberOfEpoch = tk.IntVar(value=10)  # Default value of 10
        self.yourSubject = tk.StringVar(value="Project_Name")
        self.yourPatience = tk.IntVar()
        self.yourStopPatience = tk.IntVar()
        self.yourLearningRate = tk.StringVar(value="0")
        self.yourFactor = tk.StringVar(value="0")
        self.yourFreeze = tk.BooleanVar()
        self.yourBatchSize = tk.StringVar(value="0")
        self.flattenLayer = tk.BooleanVar()
        #self.addLayer = tk.StringVar(value="0")
        #self.addLayer2 = tk.StringVar(value="0")

        self.input_height = ctk.IntVar(value=224)
        self.input_width = ctk.IntVar(value=224)


        # Main page frame
        self.create_model_main_page = ctk.CTkFrame(self, fg_color="gray30", corner_radius=0)
        self.create_model_main_page.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Left and right panels
        self.left_panel = ctk.CTkFrame(self.create_model_main_page, fg_color='gray40', corner_radius=0)
        self.left_panel.place(relx=0, rely=0, relwidth=0.60, relheight=1, anchor="nw")

        self.right_panel = ctk.CTkFrame(self.create_model_main_page, fg_color="gray50", corner_radius=0)
        self.right_panel.place(relx=0.60, rely=0, relwidth=0.40, relheight=1, anchor="nw")

        self.basicSettings = ctk.CTkLabel(self.left_panel, text="Basic Settings", fg_color="gray65", text_color="white")
        self.basicSettings.place(relx=0.02, rely=0.02, relwidth=0.35, relheight=0.05, anchor="nw")

        self.advancedSettings = ctk.CTkLabel(self.left_panel, text="Advanced Settings", fg_color="gray65", text_color="white")
        self.advancedSettings.place(relx=0.425, rely=0.02, relwidth=0.57, relheight=0.05, anchor="nw")

        # Frame for project name
        self.frame1 = ctk.CTkFrame(self.left_panel, fg_color='gray20')
        self.frame1.place(relx=0.02, rely=0.10, relwidth=0.35, relheight=0.10, anchor="nw")

        # Project name label and entry
        self.label1 = ctk.CTkLabel(self.frame1, text="Set the name of \n your project:", text_color="white")
        self.label1.place(relx=0.05, rely=0.5, anchor="w")

        # Entry for project name
        self.nomProjet = ctk.CTkEntry(self.frame1, height=30, corner_radius=5, textvariable=self.yourSubject)
        self.nomProjet.place(relx=0.48, rely=0.5, anchor="w", relwidth=0.48)

        # Bind validation to ensure only valid text is accepted
        self.nomProjet.bind("<FocusOut>", self.validate_nomProj_input)
        self.nomProjet.bind("<Return>", self.validate_nomProj_input)

        # Choosing folders
        self.training = choisirDossier(self.left_panel, textLabel="1) Load your training folder",
                                       textBouton="Choose",
                                       color="gray30",
                                       bouton_color="#48b748",
                                       command=self.open_training_folderGENERATE)
        self.training.place(relx=0.02, rely=0.25, relwidth=0.35, relheight=0.1, anchor="nw")

        self.validation = choisirDossier(self.left_panel, textLabel="2) Load your validation folder",
                                         textBouton="Choose",
                                         color="gray30",
                                         bouton_color="#48b748",
                                         command=self.open_validate_folderGENERATE)
        self.validation.place(relx=0.02, rely=0.36, relwidth=0.35, relheight=0.1, anchor="nw")

        self.output_model = choisirDossier(self.left_panel, textLabel="3) Choose where to save your model",
                                           textBouton="Choose",
                                           color="gray30",
                                           bouton_color="#48b748",
                                           command=self.open_output_folderGENERATE)
        self.output_model.place(relx=0.02, rely=0.47, relwidth=0.35, relheight=0.1, anchor="nw")

        # Epochs label and entry
        self.labelEpoch = ctk.CTkLabel(self.left_panel, text="Choose the number of Epochs",
                                       text_color="white")
        self.labelEpoch.place(relx=0.02, rely=0.62, anchor="nw")
        # Function to create the entry box
        self.Epochboite = self.create_entry_box(variable_name=self.yourNumberOfEpoch, parent=self.left_panel, width=5)
        self.Epochboite.place(relx=0.28, rely=0.62, relwidth=0.08, anchor="nw")

        # Entry for number of epochs
        self.Epochboite = ctk.CTkEntry(self.left_panel, height=30, width=100,
                                       textvariable=self.yourNumberOfEpoch)

        # validate the project name
        self.Epochboite.bind("<FocusOut>", self.validate_epoch_input)
        self.Epochboite.bind("<Return>", self.validate_epoch_input)

        # Initial options dictionary to store selected options
        self.selected_options = {}
        # Creating an option menu with predefined options
        model_options = ['MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionV3', 'NASNetMobile', 'NASNetLarge', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'VGG16', 'VGG19', 'Xception']

        self.create_option_menu("Gen", "model_selection",
                                model_options,
                                "Select a Base Model", self.left_panel, rel_x=0.02, rel_y=0.70)
        print(self.get_selected_option("Gen", "model_selection"))

        self.icon_image = "images_interface/point_dinterrogation.png"
        self.tooltip_image = "images_interface/tableau_modeles.png"
        self.create_icon_with_image_tooltip(self.left_panel, relx=0.38, rely=0.70, icon_image_path=self.icon_image, tooltip_image_path=self.tooltip_image)


        # Patience label
        self.labelPatience = ctk.CTkLabel(self.left_panel, text="Patience (in number of epoch(s))",
                                       text_color="white")
        self.labelPatience.place(relx=0.43, rely=0.1, anchor="nw")
        #Patience choice
        self.patience = ctk.CTkComboBox(
            self.left_panel,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            variable=self.yourPatience  # Bind the variable to the ComboBox
        )
        self.patience.set("1")
        self.patience.place(relx=0.85, rely=0.1, relwidth=0.1, anchor="nw")

        # Example: Creating icon with tooltip
        self.create_icon_with_tooltip(self.left_panel, relx=0.97, rely=0.1,
                                      tooltip_text="Patience is the number of epochs with no improvement to wait before reducing the learning rate by your factor.")


        # Stop Patience Label
        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.StopPatience = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Stop patience (in number of epoch(s)) \n (patience < x < epoch)",
            tooltip_text=
            "Stop patience is the number of epochs with no improvement to wait before stopping the training. This reduces the time needed to train the model and the risk of overfitting.",
            relx=0.43,
            rely=0.15,
            default_value=3
        )


        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.yourLearningRate = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Initial Learning rate (0 < x < 1)",
            tooltip_text="This is the initial value of the learning rate (usually 0.02). The learning rate represents the ability to change the strength of the connection between the neurons (the higher the value, the more changes).",
            relx=0.43,
            rely=0.20,
            is_float=True,
            default_value=0.01
        )

        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.yourFactor = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Learning rate reducing factor (0 < x < 1)",
            tooltip_text="This is the factor by which the initial learning rate will be multiplied, allowing for faster training and achieving finer accuracy during the last epochs.",
            relx=0.43,
            rely=0.25,
            is_float=True,
            default_value=0.5
        )


        # yourFreeze label
        self.yourFreezeLabel = ctk.CTkLabel(self.left_panel, text="Train the Base Model",
                                        text_color="white")
        self.yourFreezeLabel.place(relx=0.43, rely=0.30, anchor="nw")

        # tooltip base model
        self.create_icon_with_tooltip(self.left_panel, relx=0.97, rely=0.30,
                                      tooltip_text="If not checked, the weights of the parameters of the base model are frozen, allowing only the flatten and/or additional layers to be trained. If checked, the model training will be longer but more accurate.")
        # Your Freeze
        self.yourFreezeBox = self.create_checkbox(
            variable_name=self.yourFreeze,
            parent=self.left_panel,
            relx=0.88, rely=0.30
        )

        # Batch Size Label
        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.batchSize = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Batch size",
            tooltip_text="The batch size is the number of images in each batch used during training. It is recommended to calculate it by multiplying the number of classes by 10. However, you can reduce this number if you do not have enough memory (RAM).",
            relx=0.43,
            rely=0.35,
            default_value=64
        )



        # tooltip flatten
        self.create_icon_with_tooltip(self.left_panel, relx=0.97, rely=0.40,
                                      tooltip_text="The flatten layer transforms a multidimensional input into a unidimensional output. You can use it to transform the last tridimensional convolutional layer into a unidimensional layer, which is needed for classifying images into different classes. If you don’t use the flatten layer, another option called global average pooling will be used.")

        # flattenLayer label
        self.flattenLabel = ctk.CTkLabel(self.left_panel, text="Add a flatten layer",
                                            text_color="white")
        self.flattenLabel.place(relx=0.43, rely=0.40, anchor="nw")


        self.flattenCheckBox = self.create_checkbox(
            variable_name=self.flattenLayer,
            parent=self.left_panel,
            relx=0.88, rely=0.40)

        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.dropout_var = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Set Dropout",
            tooltip_text="Dropout temporarily and randomly disables a percentage of the neurons (and their connections) in the previous layer. This optimizes the model and decreases overfitting. Use a value between 0 and 1 to set this percentage.",
            relx=0.43,
            rely=0.45,
            is_float=True,
            default_value=0.2
        )

        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.addLayer_var = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Add one unidimensional layer of size :",
            tooltip_text="Add one unidimensional layer before the output layer. It is usually set between 1000 to 4000 for the best results on the ImageNet dataset (with 1000 classes in the output).",
            relx=0.43,
            rely=0.50,
            default_value=1000
        )

        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.addLayer2_var = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Add another unidimensional layer of size :",
            tooltip_text="Add another unidimensional layer before the output layer.",
            relx=0.43,
            rely=0.55,
            default_value=256
        )

        # Appel de la fonction pour créer le label, l'entry et le tooltip
        self.addLayer3_var = self.create_labeled_entry(
            parent_frame=self.left_panel,
            label_text="Add another unidimensional layer of size :",
            tooltip_text="Add another unidimensional layer before the output layer.",
            relx=0.43,
            rely=0.60
        )

        # Train model button
        self.TrainModel = ctk.CTkButton(
            self.left_panel,
            text="4) Train your model",
            fg_color="green",
            text_color="white",
            command=lambda: threading.Thread(target=self.run_TrainModel).start()  # Run in a separate thread
        )
        self.TrainModel.place(relx=0.1, rely=0.9, anchor="nw", relwidth=0.8)

        # Explanation text box with rounded corners and improved readability
        # Define custom fonts
        arial_font = font.Font(family="Arial", size=14, weight="normal", slant="italic")
        courier_font = font.Font(family="Courier", size=12, weight="bold")

        # Explanation text box with rounded corners and improved readability
        self.explicationGenerate = ctk.CTkTextbox(
            self.right_panel,
            fg_color="#7f8c8d",
            corner_radius=20,
            font=("Arial", 14, "italic"),
            text_color="white",
            wrap="word"
        )

        # Insert explanatory content
        text_content = """This module enables users to configure and train an image classification model effectively, offering both basic and advanced customization options to suit various levels of expertise and project requirements.

Step-by-step guide
Basic Usage: To train a model with minimal setup (left panel green buttons):
    1) Load the folder containing your training dataset (e.g., the "training" folder generated during the augmentation step).

    2) Load the folder containing your validation dataset.

    3) Choose the folder where the trained model will be saved

    4) Start the training process. (Note: File paths must not contain accents or special characters; otherwise, the training will fail to start).
    
Outputs : A .keras file containing your trained model and a .txt file containing class information will be generated in the output folder. These two files will be used in the "Annotate and Sort" step.
________________________________________
In the basic settings, users can also:
	• Assign a project name, which will be included in the saved model's filename (.keras file).
	• Specify the number of epochs for training. For more information, see the "Specify the Number of Epochs for Training" paragraph below.
	• Define the input image dimensions for the model (these can differ from the original image resolution). For more information, see the "Define the Input Image Dimensions for the Model" paragraph below.
	• Choose a base model from a range of pre-trained architectures, including:
MobileNetV2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, NASNetMobile, NASNetLarge, ResNet50V2, ResNet101V2, ResNet152V2, VGG16, VGG19, and Xception.

________________________________________
Specify the Number of Epochs for Training
An epoch represents one complete pass of the entire training dataset through the model during training. Specifying the number of epochs determines how many times the model will see the entire dataset and update its parameters based on the errors it encounters.
	• Importance of the Number of Epochs:
    	    - Too few epochs may result in underfitting, where the model fails to learn enough from the training data and performs poorly on new, unseen data.
    	    - Too many epochs may lead to overfitting, where the model learns patterns specific to the training data, reducing its ability to generalize to new data.
	• How to Choose the Number of Epochs:
Start with a moderate value (e.g., 10–20 epochs) for an initial test. Monitor the model’s performance on the validation dataset. If the validation accuracy continues to improve significantly, you can increase the number of epochs. Conversely, if the validation accuracy plateaus or begins to decline, it may be a sign to stop training or adjust other parameters.
________________________________________
Define the Input Image Dimensions for the Model
Image dimensions refer to the width and height of the images fed into the model. While the original resolution of your dataset images can vary, the model requires a uniform input size to process them.
	• Why Resize Images for Training?
Neural networks require fixed input dimensions to ensure consistent processing. Resizing all images to a uniform size avoids computational errors and ensures compatibility with the chosen model architecture.
	• How to Define Input Dimensions:
You can specify the desired width and height (e.g., 224x224 pixels). The choice of dimensions often depends on the selected base model:
    	    - Smaller dimensions (e.g., 128x128) reduce computational load and training time, but they may sacrifice some detail in the images.
    	    - Larger dimensions (e.g., 299x299) retain more detail, which can improve model accuracy for complex tasks, though at the cost of increased memory and processing requirements.
	• Impact of Dimensions on Performance:
The model’s ability to recognize fine details, such as subtle patterns in high-resolution images, depends on these dimensions. For datasets with small or intricate features, selecting a higher resolution may enhance performance. However, for simpler datasets, lower resolutions may suffice, enabling faster training.


________________________________________
Going Further
To improve your model or adapt it to specific needs, you can customize advanced settings. Each parameter in the advanced settings is explained below:

- Patience Parameters:
    	• Patience: Defines the number of epochs with no improvement in the validation metric (e.g., accuracy or loss) before the model's training process considers adjusting other parameters or stopping. A higher patience value allows the model to train for longer without interruption.
    	• Stopping Patience: If the validation metric shows no improvement after a defined number of epochs, training will terminate early to prevent overfitting or wasting resources.

- Learning Rate Configuration:
    	• Initial Learning Rate: Sets the starting step size for updating the model's weights. A lower learning rate ensures precise adjustments, while a higher rate speeds up the process but risks skipping optimal solutions.
    	• Learning Rate Reducing Factor: Determines the scale at which the learning rate decreases when the model’s performance stagnates. For example, a factor of 0.1 will reduce the learning rate to 10% of its current value.

- Trainable Base Model:
    	• Enabling this option allows the pre-trained base model's weights to be fine-tuned during training. This is particularly useful for domain-specific datasets, but it requires more computational resources. Disabling this option keeps the base model frozen, only training the custom layers.

- Batch Size:
    	• Specifies the number of images processed together in a single training step. For smaller image resolutions, increasing the batch size speeds up training. However, higher batch sizes require more memory.

- Custom Layers Configuration:
    	• Users can choose to add: 
        	    - A Flatten Layer: Converts the base model’s multi-dimensional output into a one-dimensional format suitable for dense layers.
        	    - One to three Dense (Fully Connected) Layers: These layers consist of a set number of neurons, which users can define. They are used to learn complex patterns and representations from the base model’s output.
        	    - Dropout Layer: Introduces a regularization technique by randomly "dropping out" a fraction of neurons during training to prevent overfitting. The dropout rate determines the percentage of neurons to deactivate.


________________________________________
Model Architecture Overview
1.	Base Model:
The pre-trained architecture forms the foundation of your model. It extracts features from images using convolutional layers optimized for specific tasks during competitions. For example, MobileNetV2 is particularly CPU-efficient, while ResNet variants often excel in GPU-based environments.

2.	Custom Layers:
These layers are appended to the base model to adapt it to your dataset. The custom layers consist of:
	• A bridging connection from the base model to fully connected layers.
	• Fully connected layers for high-level pattern learning.
	• A final output layer, generated automatically based on the number of classes in your dataset."""
        self.explicationGenerate.insert("1.0", text_content)
        self.explicationGenerate.place(relx=0.5, rely=0.42, anchor="center", relwidth=0.95,
                                       relheight=0.8)  # Centered in the right panel

        # Ajouter un encadré pour afficher le contenu du terminal
        self.terminal_textbox = ctk.CTkTextbox(self.right_panel, wrap="word")
        self.terminal_textbox.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.99, relheight=0.15)
        # Configuration de la police monospace
        monospace_font = ctk.CTkFont(family="Courier", size=12)
        self.terminal_textbox.configure(font=monospace_font)  # Appliquer la police monospace
        # Optionnel : définir du texte par défaut
        self.terminal_textbox.insert("1.0", "Code output...")
        # Initialize stdout redirector without the callback since we don't need display_images
        self.stdout_redirector = StdoutRedirector(
            callback=None,  # Pas besoin de callback pour juste afficher le terminal
            callback_args=None,
            complete_message="",  # Pas de message complet nécessaire ici
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )



        # Height and Width
        self.dimension_frame = self.create_dimension_frame(
            parent=self.left_panel,
            relx=0.18,
            rely=0.82,
            relwidth=0.3,
            relheight=0.12,
            label_text="Set your height and width",
        )


        # Commencez la redirection du stdout vers le terminal_textbox
        self.stdout_redirector.start()

    # Function to handle model training in a separate thread
    def run_TrainModel(self):
        # Start redirecting stdout to the terminal_textbox
        self.stdout_redirector.start()

        # Run the training function
        train_model(
            yoursubject=str(self.yourSubject.get()),
            yournumberofepochs=int(self.yourNumberOfEpoch.get()),
            training_folder=self.training_folder_pathGENERATE,
            validation_folder=self.validate_folder_pathGENERATE,
            save_dir=self.output_folder_pathGENERATE,
            yourmodel_name=str(self.get_selected_option("Gen", "model_selection")),
            yourheight=self.input_height.get(),
            yourwidth=self.input_width.get(),
            yourpatience=self.patience.get(),
            yourstop_patience=self.StopPatience.get(),
            yourlearningrate=self.yourLearningRate.get(),
            yourfactor=self.yourFactor.get(),
            yourbatch_size=self.batchSize.get(),
            yoursizeofadditionallayer1=self.addLayer_var.get(),
            yoursizeofadditionallayer2=self.addLayer2_var.get(),
            yoursizeofadditionallayer3=self.addLayer3_var.get(),
            YourDropout=self.dropout_var.get(),
            Useflattenlayer=self.flattenLayer.get(),
            yourfreeze=self.yourFreeze.get()
        )



        # Train model button
        #self.TrainModel = ctk.CTkButton(self.left_panel, text="Train your model", fg_color="green",
        #                               text_color="white",
        #                               command=lambda: train_model(
        #                                   yoursubject=str(self.yourSubject.get()),
        #                                   yournumberofepochs=int(self.yourNumberOfEpoch.get()),
        #                                   training_folder=self.training_folder_pathGENERATE,
        #                                   validation_folder=self.validate_folder_pathGENERATE,
        #                                    save_dir=self.output_folder_pathGENERATE
        #                                ))
        #self.TrainModel.place(relx=0.2, rely=0.93, anchor="nw", relwidth=0.6)

    def create_icon_with_tooltip(self, parent_frame, relx, rely, tooltip_text):

        # Create the icon label in the specified parent frame at the given position
        icon_label = ctk.CTkLabel(parent_frame, image=self.image_icon, text="")  # Image in label with no text
        icon_label.place(relx=relx, rely=rely, anchor="nw")

        # Attach the tooltip to the icon label
        Tooltip(icon_label, tooltip_text)

    def create_icon_with_image_tooltip(self, parent_frame, relx, rely, icon_image_path, tooltip_image_path):
        # Load the icon image for the button
        icon_image = Image.open(icon_image_path)
        icon_image = icon_image.resize((25, 25))  # Resize if needed
        icon_photo = ImageTk.PhotoImage(icon_image)

        # Create the icon label in the specified parent frame at the given position
        icon_label = ctk.CTkLabel(parent_frame, image=icon_photo, text="")  # Image in label with no text
        icon_label.place(relx=relx, rely=rely, anchor="nw")

        # Attach the image tooltip to the icon label
        ImageTooltip(icon_label, tooltip_image_path, position="right", x_offset=10, y_offset=10)

        # Make sure the image is accessible during the program's lifetime
        icon_label.image = icon_photo



    def create_entry_box(self, parent, variable_name, width=100):

        # Create the entry box
        entry_box = ctk.CTkEntry(parent, height=30, width=width, textvariable=variable_name)

        # Bind events to save the entry value
        entry_box.bind("<FocusOut>", lambda event: self.save_value(variable_name))
        entry_box.bind("<Return>", lambda event: self.save_value(variable_name))

        # Return the created entry box

        return entry_box

    def save_value(self, variable_name):

        # Access the current value in the entry and store it (already done by the textvariable)
        current_value = variable_name.get()
        print(f"Value saved: {current_value}")
        # Additional saving logic can be implemented here if needed

    def create_checkbox(self, variable_name, parent, relx, rely):

        # Create the checkbox without a text label
        checkbox = ctk.CTkCheckBox(
            parent,
            variable=variable_name,
            text="",        # No text, so no extra space for label
            width=20,       # Set to the same size as the checkbox
            height=20       # Set to the same size as the checkbox
        )
        checkbox.place(relx=relx, rely=rely, anchor="nw")

        # Bind the variable change to print the value using a lambda
        variable_name.trace_add("write", lambda *args: self.print_boolean_value(variable_name))

        # Return the created checkbox
        return checkbox

    def print_boolean_value(self, variable_name, *args):

        print(f"Checkbox state: {variable_name.get()} (type: {type(variable_name.get())})")
        # Stop redirecting stdout after training completes
        self.stdout_redirector.stop()

    def validate_nomProj_input(self, event):
        text = str(self.yourSubject.get())
        if text:
            self.yourSubject.set(text)
            print(f"Project name set to: {text}")
        else:
            print("Project name is empty. Resetting to default value.")
            self.yourSubject.set("Project_Name")  # Reset to default value if needed

    def validate_epoch_input(self, event):
        text = self.Epochboite.get().strip()  # Get and strip any extra whitespace
        if text:
            try:
                # Try to convert the text to an integer
                value = int(text)
                self.yourNumberOfEpoch.set(value)
                print(f"Epochs set to: {self.yourNumberOfEpoch.get()}")
            except ValueError:
                # If conversion fails, reset to default value and show an error message
                self.yourNumberOfEpoch.set(10)  # Reset to default value
                print("Invalid input. Resetting to default value: 10")
        else:
            # If the input is empty, reset to default value
            self.yourNumberOfEpoch.set(10)  # Reset to default value
            print("Input is empty. Resetting to default value: 10")

    def open_training_folderGENERATE(self):
        # Open a file dialog to select the import folder for Augment page
        selected_folder_importGENERATE = os.fsdecode(tk.filedialog.askdirectory(title="Select Import Folder for Training"))
        if selected_folder_importGENERATE:
            # Update the import folder path variable for Augment page
            self.training_folder_pathGENERATE = selected_folder_importGENERATE
            print(f"Selected import folder path for Training: {self.training_folder_pathGENERATE}")

            # Update the label text to show the selected folder
            #self.import_folder_label.configure(text=self.training_folder_pathGENERATE)

    def open_validate_folderGENERATE(self):
        # Open a file dialog to select the import folder for Augment page
        selected_folder_importGENERATE = os.fsdecode(tk.filedialog.askdirectory(title="Select Import Folder for Validation"))

        if selected_folder_importGENERATE:
            # Update the import folder path variable for Augment page
            self.validate_folder_pathGENERATE = selected_folder_importGENERATE
            print(f"Selected import folder path for validation: {self.validate_folder_pathGENERATE}")

            # Update the label text to show the selected folder
            #self.import_folder_label.configure(text=self.validate_folder_pathGENERATE)


    def open_output_folderGENERATE(self):
        # Open a file dialog to select the import folder for Augment page
        selected_folder_importGENERATE = os.fsdecode(tk.filedialog.askdirectory(title="Select Output Folder to save your trained model"))

        if selected_folder_importGENERATE:
            # Update the import folder path variable for Augment page
            self.output_folder_pathGENERATE = selected_folder_importGENERATE
            print(f"Selected export folder path to save the model: {self.output_folder_pathGENERATE}")

            # Update the label text to show the selected folder
            #self.import_folder_label.configure(text=self.output_folder_pathGENERATE)

    def create_page_Visualize(self):

        # Default values
        self.input_height = ctk.StringVar(value="224")
        self.input_width = ctk.StringVar(value="224")

        self.chosen_file_VISUALISE = "Select a file"
        self.export_folder_path_VISUALISE = "Select a folder"


        #remise à zero des images et classification report à l'ouverture de la page
        # Répertoire des images
        base_dir = os.path.dirname(__file__)
        images_dir = os.path.join(base_dir, 'images_interface')
        # Chemins source et destination
        source = os.path.join(images_dir, "confusion_matrix_vide.png")
        destination = os.path.join(images_dir, "confusion_matrix.png")
        # Copier le fichier et renommer
        shutil.copyfile(source, destination)
        # Chemins source et destination
        source = os.path.join(images_dir, "errors_by_class_vide.png")
        destination = os.path.join(images_dir, "errors_by_class.png")
        # Copier le fichier et renommer
        shutil.copyfile(source, destination)
        # Chemins source et destination
        source = os.path.join(images_dir, "classification_report_vide.txt")
        destination = os.path.join(images_dir, "classification_report.txt")
        # Copier le fichier et renommer
        shutil.copyfile(source, destination)



        self.page_Visualize = ctk.CTkFrame(self, fg_color="gray60", corner_radius=0)
        self.page_Visualize.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        self.left_panel = ctk.CTkFrame(self.page_Visualize, fg_color='gray80', corner_radius=0)
        self.left_panel.place(relx=0, rely=0, relwidth=0.25, relheight=1, anchor="nw")

        self.right_panel = ctk.CTkFrame(self.page_Visualize, fg_color="gray90", corner_radius=0)
        self.right_panel.place(relx=0.25, rely=0, relwidth=0.75, relheight=1, anchor="nw")

        # Create a canvas and scrollbar for the right panel
        self.canvas = ctk.CTkCanvas(self.right_panel, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ctk.CTkScrollbar(self.right_panel, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas for placing images
        self.image_frame = ctk.CTkFrame(self.canvas, fg_color="white")
        self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

        # Bind the canvas size to the image frame
        self.image_frame.bind("<Configure>",
                              lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Debug statement to ensure self.image_frame is initialized correctly
        if not hasattr(self, 'image_frame'):
            print("Error: image_frame is not defined.")
        #else:
            #print("Success: image_frame has been defined properly.")





        # Load the model
        self.import_data = ctk.CTkButton(self.left_panel, text="1) Load your model",
                                         command=self.select_file_VISUALISE)
        self.import_data.place(relx=0.5, rely=0.05, anchor="center", relwidth=0.8, relheight=0.08)

        # Label to display the chosen file
        self.chosen_file_label_VISUALISE = ctk.CTkLabel(self.left_panel, text=self.chosen_file_VISUALISE,
                                                        text_color="white")
        self.chosen_file_label_VISUALISE.place(relx=0.5, rely=0.12, anchor="center", relwidth=0.8,
                                               relheight=0.06)

        # Select export folder
        self.exportVISUALISE = ctk.CTkButton(self.left_panel, text="2) Select dataset folder",
                                             command=self.open_export_folder_VISUALISE)
        self.exportVISUALISE.place(relx=0.5, rely=0.20, anchor="center", relwidth=0.8, relheight=0.08)

        # Label to show selected export folder path
        self.export_folder_label_VISUALISE = ctk.CTkLabel(self.left_panel,
                                                          text=self.export_folder_path_VISUALISE,
                                                          text_color="white")
        self.export_folder_label_VISUALISE.place(relx=0.5, rely=0.27, anchor="center", relwidth=0.8,
                                                 relheight=0.06)

        # Dimension settings frame
        self.dimension_frame = ctk.CTkFrame(self.left_panel, fg_color="gray40")
        self.dimension_frame.place(relx=0.5, rely=0.4, relwidth=0.8, relheight=0.2, anchor="center")

        # Configure rows and columns to adjust the size dynamically
        self.dimension_frame.grid_rowconfigure(0, weight=1)  # Main label row
        self.dimension_frame.grid_rowconfigure(1, weight=1)  # Height row
        self.dimension_frame.grid_rowconfigure(2, weight=1)  # Width row
        self.dimension_frame.grid_columnconfigure(0, weight=1)  # Left side labels
        self.dimension_frame.grid_columnconfigure(1, weight=1)  # Right side entries

        # Main label with wrap enabled to prevent overflow
        self.label = ctk.CTkLabel(self.dimension_frame, text="3) Set the image size used for your model",
                                  font=("Arial", 14), text_color="white", wraplength=200, anchor="center")
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Height label and entry
        self.height_label = ctk.CTkLabel(self.dimension_frame, text="Height:", font=("Arial", 12),
                                         text_color="white")
        self.height_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")

        self.heightbox = ctk.CTkEntry(self.dimension_frame, width=100, height=30,
                                      textvariable=self.input_height, placeholder_text="224")
        self.heightbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Width label and entry
        self.width_label = ctk.CTkLabel(self.dimension_frame, text="Width:", font=("Arial", 12),
                                        text_color="white")
        self.width_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")

        self.widthbox = ctk.CTkEntry(self.dimension_frame, width=100, height=30,
                                     textvariable=self.input_width, placeholder_text="224")
        self.widthbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Explanation text box with rounded corners and improved readability
        self.explicationVisualize = ctk.CTkTextbox(
            self.left_panel,
            fg_color="#7f8c8d",
            corner_radius=20,
            font=("Arial", 14, "italic"),
            text_color="white",
            wrap="word"
        )

        # Insert explanatory content
        text_contentVisualize = """Step-by-step guide:
        
   1) Load your trained model file in .keras format. This file contains the architecture and weights of your trained model.
   2) Select the folder containing the dataset for testing. Typically, this is the "test" folder created during the augmentation step, containing images unseen during training.
   3) Configure the target size for testing by specifying the resolution used during training. This information is included in the model filename, immediately following the word "resolution."
   4) Start the testing process to evaluate the model's predictions.
   5) Review the outputs, which include:
        -Saved graphs visualizing metrics like accuracy and loss over the test dataset.
        -A classification report summarizing key performance metrics such as precision, recall, F1-score, and AUC ROC for each class.
        -A .txt file containing class information will be generated in the output folder. This .txt file is essential for the subsequent “Annotate and sort” step.



This module enables you to evaluate the performance of your trained model on a dataset that was not used during training. By generating visual outputs and detailed classification reports, it plays a crucial role in assessing the effectiveness and generalizability of your model while offering valuable insights into potential areas for improvement.
"""
        self.explicationVisualize.insert("1.0", text_contentVisualize)
        self.explicationVisualize.place(relx=0.5, rely=0.61, anchor="center", relwidth=0.95,
                                       relheight=0.20)  # Centered in the right panel

        # Visualize button
        self.explicationVisualize = ctk.CTkButton(self.left_panel, text="4) Visualize your results",
                                       command=self.start_visualization, fg_color="#e74c3c")
        self.explicationVisualize.place(relx=0.5, rely=0.74, anchor="center")

        # Output button Visualize
        self.open_outputFolderVisualize = ctk.CTkButton(self.left_panel, text="5) Open output folder",
                                                   command=lambda: os.startfile(self.export_folder_path_VISUALISE))
        self.open_outputFolderVisualize.place(relx=0.5, rely=0.79, anchor="center")

        # Initial display of images
        self.display_images(self.image_frame)

        # Ajouter un encadré pour afficher le contenu du terminal
        self.terminal_textbox = ctk.CTkTextbox(self.left_panel, wrap="word")
        self.terminal_textbox.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.8, relheight=0.15)
        # Optionnel : définir du texte par défaut
        self.terminal_textbox.insert("1.0", "Code output...")
        self.stdout_redirector = StdoutRedirector(
            callback=None,  # Pas besoin de callback pour juste afficher le terminal
            callback_args=None,
            complete_message="",  # Pas de message complet nécessaire ici
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )

        # Commencez la redirection du stdout vers le terminal_textbox
        self.stdout_redirector.start()

    def create_dimension_frame(self, parent, relx, rely, relwidth, relheight, label_text):
        # Assurez-vous que self.input_height et self.input_width sont définis comme des variables Tkinter
        self.input_height = tk.IntVar(value=224)  # Ou DoubleVar() si nécessaire
        self.input_width = tk.IntVar(value=224)

        # Create the frame
        frame = ctk.CTkFrame(parent, fg_color="gray60")
        frame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight, anchor="center")

        # Main label
        main_label = ctk.CTkLabel(frame, text=label_text, font=("Arial", 14), text_color="white")
        main_label.place(relx=0.5, rely=0.1, anchor="center")

        # Height label and entry
        height_label = ctk.CTkLabel(frame, text="Height:", font=("Arial", 10), text_color="white")
        height_label.place(relx=0.25, rely=0.4, anchor="center")

        height_entry = ctk.CTkEntry(frame, width=80, height=25, textvariable=self.input_height)
        height_entry.place(relx=0.5, rely=0.4, anchor="center")

        # Event to update input_height when the entry field changes
        def update_height_var(*args):
            self.input_height.set(height_entry.get())  # Update the input_height variable with the current value

        # Bind the update function to changes in the height_entry
        height_entry.bind("<KeyRelease>", update_height_var)  # Update on key release
        height_entry.bind("<FocusOut>", update_height_var) # Mettre à jour lors de la perte du focus

        # Width label and entry
        width_label = ctk.CTkLabel(frame, text="Width:", font=("Arial", 10), text_color="white")
        width_label.place(relx=0.25, rely=0.6, anchor="center")

        width_entry = ctk.CTkEntry(frame, width=80, height=25, textvariable=self.input_width)
        width_entry.place(relx=0.5, rely=0.6, anchor="center")

        # Event to update input_width when the entry field changes
        def update_width_var(*args):
            self.input_width.set(width_entry.get())  # Update the input_width variable with the current value

        # Mettre à jour lors de la saisie de texte (touche relâchée) ou lorsque l'entrée perd le focus
        width_entry.bind("<KeyRelease>", update_width_var)  # Mettre à jour lors d'une touche relâchée
        width_entry.bind("<FocusOut>", update_width_var)  # Mettre à jour lors de la perte du focus

        return frame
    def start_visualization(self):
        # Initialize stdout redirector with the display_images method
        self.stdout_redirector = StdoutRedirector(
            self.display_images,
            callback_args=[self.right_panel],
            complete_message="Visualization complete",
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )
        # Start redirecting stdout
        self.stdout_redirector.start()

        # Perform visualization in a separate thread to keep the UI responsive
        threading.Thread(target=self.run_visualize).start()

    def run_visualize(self):
        # Call the visualize function
        visualize(
            int(self.input_height.get()),
            int(self.input_width.get()),
            self.chosen_file_VISUALISE,
            self.export_folder_path_VISUALISE
        )

        # After visualization completes, stop redirecting stdout
        self.stdout_redirector.stop()

        # Optionally, you might want to display images immediately after
        self.display_images(self.image_frame)

    def display_images(self, panel):
        # Clear previous content in the image panel
        for widget in panel.winfo_children():
            widget.destroy()

        # List of image paths and their sizes
        base_dir = os.path.dirname(__file__)  # current script
#        images_dir = os.path.join(base_dir, 'images_interface')
        if self.export_folder_label_VISUALISE.cget("text") != "Select a folder":
            images_dir = os.path.join(self.export_folder_label_VISUALISE.cget("text"))
        else:
            images_dir = os.path.join(base_dir, 'images_interface')

        images = [
            {"path": os.path.join(images_dir, "confusion_matrix.png"),
             "size": (650, 650)},
            {"path": os.path.join(images_dir, "errors_by_class.png"),
             "size": (700, 250)}
        ]

        # Image labels
        image_labels = []

        for i, img_info in enumerate(images):
            try:
                # Charger l'image
                img = Image.open(img_info["path"])

                # Obtenir la taille originale de l'image
                original_width, original_height = img.size
                max_width, max_height = img_info["size"]

                # Calculer les facteurs d'échelle pour la largeur et la hauteur
                width_ratio = max_width / original_width
                height_ratio = max_height / original_height

                # Choisir le plus petit facteur d'échelle pour conserver les proportions
                scale_factor = min(width_ratio, height_ratio)

                # Calculer les nouvelles dimensions de l'image
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                # Redimensionner l'image en gardant les proportions
                img = img.resize((new_width, new_height))

                # Convertir l'image PIL en CTkImage avec les nouvelles dimensions
                ctk_image = ctk.CTkImage(light_image=img, size=(new_width, new_height))

                # Créer un CTkLabel pour afficher l'image
                img_label = ctk.CTkLabel(panel, image=ctk_image, text="")  # Pas de texte, juste l'image
                img_label.image = ctk_image  # Garder une référence pour éviter la collecte des ordures (garbage collection)
                image_labels.append(img_label)

            except Exception as e:
                print(f"Error loading image {img_info['path']}: {e}")

        # Arrange images using grid
            # Vérifier que la liste 'images' contient exactement 2 chemins et que chaque fichier existe
        if len(images) == 2 and all(os.path.exists(img_info["path"]) for img_info in images):
            image_labels[0].grid(row=0, column=0, padx=5, pady=5, sticky="n")
            image_labels[1].grid(row=1, column=0, padx=5, pady=5, sticky="n")
            #image_labels[2].grid(row=1, column=1, padx=(5, 10), pady=10, sticky="ew")

        # Adjust the column weights to ensure proper centering
        panel.grid_columnconfigure(0, weight=1)
        #panel.grid_columnconfigure(1, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Chemin du fichier rapport
        base_dir = os.path.dirname(__file__)
        #images_dir = os.path.join(base_dir, 'images_interface')
        report_path = os.path.join(images_dir, 'classification_report.txt')
        # Ajouter un widget texte à défilement pour afficher le rapport avec une taille précise
        texte = scrolledtext.ScrolledText(self.right_panel, wrap=tk.NONE, width=40, height=60, borderwidth=1, relief="solid")
        # Fixer manuellement la taille de l'encadré texte à 300x1000
        texte.place(relx=0.68, rely=0, relwidth=0.3, relheight=0.7, anchor="nw")  # Taille définie manuellement
        # Ajouter la barre de défilement horizontale (déjà incluse dans ScrolledText mais désactivée si wrap est autre chose que NONE)
        #texte.config(xscrollcommand=texte.xview, yscrollcommand=texte.yview)
        # Personnaliser la police et la taille du texte
        #police = font.Font(family="Helvetica", size=12)  # Exemple : Police Helvetica, taille 12
        #texte.configure(font=police)
        # Lire le fichier et insérer son contenu
        with open(report_path, 'r') as report_file:
            contenu = report_file.read()
            texte.insert(tk.INSERT, contenu)
        # Positionner le texte dans la fenêtre
        #texte.pack(padx=20, pady=20)



    def select_file_VISUALISE(self):
        file_path = tk.filedialog.askopenfilename(title="Select a file", filetypes=(("All files", "*.*"),))
        if file_path:
            self.chosen_file_VISUALISE = file_path
            self.chosen_file_label_VISUALISE.configure(text=file_path)

    def open_export_folder_VISUALISE(self):
        folder_path = tk.filedialog.askdirectory(title="Select a folder")
        if folder_path:
            self.export_folder_path_VISUALISE = folder_path
            self.export_folder_label_VISUALISE.configure(text=folder_path)

    # Add getter methods to retrieve the paths
    # Corrected pass_variable function to properly return paths
    def pass_variable(self, path):
        return path

    def set_fill_mode(self, mode):
        self.fill_mode.set(mode)

    def create_use_model(self):

        self.import_model_pathUSE = ""
        self.import_classInfoUSE = ""

        self.page_UseModel = ctk.CTkFrame(self, fg_color='gray60')
        self.page_UseModel.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Left and right panels
        self.left_panelUSE = ctk.CTkFrame(self.page_UseModel, fg_color='gray40', corner_radius=0)
        self.left_panelUSE.place(relx=0, rely=0, relwidth=0.5, relheight=1, anchor="nw")

        self.right_panelUSE = ctk.CTkFrame(self.page_UseModel, fg_color="gray50", corner_radius=0)
        self.right_panelUSE.place(relx=0.5, rely=0, relwidth=0.5, relheight=1, anchor="nw")

        self.charger_model = ctk.CTkButton(self.left_panelUSE,
                                    text="1) Load your Model",
                                    command=lambda: self.open_chargerModel_USE(),
                                    fg_color="#2980b9",
                                    text_color="white",
                                    hover_color="#3498db",
                                    corner_radius=10
                                    )
        self.charger_model.place(relx=0.5, rely=0.10, anchor="center")

        self.charger_model_labelUSE = ctk.CTkLabel(self.left_panelUSE,
                                                   text=self.import_model_pathUSE,
                                                   text_color="white",
                                                   font=("Arial", 12)
                                                   )
        self.charger_model_labelUSE.place(relx=0.5, rely=0.15, anchor="center")

        self.charger_class_information = ctk.CTkButton(self.left_panelUSE,
                                    text="2) Load class information",
                                    command=lambda: self.open_classInfoUSE(),
                                    fg_color="#2980b9",
                                    text_color="white",
                                    hover_color="#3498db",
                                    corner_radius=10)
        self.charger_class_information.place(relx=0.5, rely=0.20, anchor="center")

        self.charger_classe_labelUSE = ctk.CTkLabel(self.left_panelUSE,
                                                    text=self.import_classInfoUSE,
                                                    text_color="white",
                                                    font=("Arial", 12)
                                                    )
        self.charger_classe_labelUSE.place(relx=0.5, rely=0.25, anchor="center")


        self.do_prediction = ctk.CTkButton(self.left_panelUSE, text="5) Do prediction", command=lambda: Annotate(model_path=self.import_model_pathUSE, dictionnary_path=self.import_classInfoUSE, input_dir=self.folder_paths["create_use_model_folder_withimagestopredict"], output_dir=self.folder_paths["create_use_model_folder_tostorepredictedimages"]), fg_color="#e74c3c")
        self.do_prediction.place(relx=0.5, rely=0.60, anchor="center")

        # Initial folder paths dictionary
        self.folder_paths = {}
        # Creating multiple import buttons with labels
        self.create_folder_import_button("create_use_model", "folder_withimagestopredict", "3) Select Folder with images to predict",
                                         self.left_panelUSE, 0.5, 0.35)
        self.create_folder_import_button("create_use_model", "folder_tostorepredictedimages",
                                         "4) Choose folder to store predicted images",
                                         self.left_panelUSE, 0.5, 0.45)

        self.open_outputFolder_USE = ctk.CTkButton(self.left_panelUSE, text="6) Open output folder", command=lambda: os.startfile(self.folder_paths["create_use_model_folder_tostorepredictedimages"]))
        self.open_outputFolder_USE.place(relx=0.5, rely=0.8, anchor="center")

        self.explicationGenerate = ctk.CTkTextbox(
            self.right_panelUSE,
            fg_color="#7f8c8d",
            corner_radius=20,
            font=("Arial", 14, "italic"),
            text_color="white",
            wrap="word"
        )

        # Insert explanatory content
        text_content = """Step-by-step guide:
    1) Load your saved model (.keras file).
    2) Load the .txt file containing the class information for which your model is trained. This file, named "your_dictionary_withXclasses.txt" (where X is the number of classes), is generated during the "Train your model" or "Evaluate performance" process.
    3) Select the folder containing the images you want to classify.
    4) Choose the folder where you want to save your classified images.
    5) Start the prediction process.
    6) Open the folder to view the predicted results (images are annotated with the first three letters of your classes, the predicted percentage for each class, and the original name of the image).

Thanks for using VisuelAIclassification.
Developed by Théo CHARNAY, Leo Zwilling and Eric Pellegrino.
You can contact us at: theo.CHARNAY@univ-amu.fr
Please cite us if you do publication using this software.

Copyright 2024 Theo CHARNAY, Leo Zwilling and Eric Pellegrino

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
                        """
        self.explicationGenerate.insert("1.0", text_content)
        self.explicationGenerate.place(relx=0.5, rely=0.41, anchor="center", relwidth=0.89,
                                       relheight=0.8)  # Centered in the right panel

        # Ajouter un encadré pour afficher le contenu du terminal
        self.terminal_textbox = ctk.CTkTextbox(self.right_panelUSE, wrap="word")
        self.terminal_textbox.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.99, relheight=0.15)
        # Optionnel : définir du texte par défaut
        self.terminal_textbox.insert("1.0", "Code output...")
        # Initialize stdout redirector without the callback since we don't need display_images
        self.stdout_redirector = StdoutRedirector(
            callback=None,  # Pas besoin de callback pour juste afficher le terminal
            callback_args=None,
            complete_message="",  # Pas de message complet nécessaire ici
            terminal_textbox=self.terminal_textbox  # Passer le terminal_textbox pour l'affichage du stdout
        )

        # Commencez la redirection du stdout vers le terminal_textbox
        self.stdout_redirector.start()

    def open_chargerModel_USE(self):
        # Open a file dialog to select the import folder for Augment page
        selected_folder_importUSE = tk.filedialog.askopenfilename(title="Select your trained model")

        if selected_folder_importUSE:
            # Update the import folder path variable for Augment page
            self.import_model_pathUSE = selected_folder_importUSE
            print(f"Selected trained model: {self.import_model_pathUSE}")

            # Update the label text to show the selected folder
            self.charger_model_labelUSE.configure(text=self.import_model_pathUSE)

    def open_classInfoUSE(self):

        selected_folder_classInfoUSE = tk.filedialog.askopenfilename(title="Select class information")

        if selected_folder_classInfoUSE:
            # Update the import folder path variable for Augment page
            self.import_classInfoUSE = selected_folder_classInfoUSE
            print(f"Selected import folder path for Augment: {self.import_classInfoUSE}")

            # Update the label text to show the selected folder
            self.charger_classe_labelUSE.configure(text=self.import_classInfoUSE)

    def create_Annotate(self):

        self.page_Annotate = ctk.CTkFrame(self, fg_color='gray60')
        self.page_Annotate.place(relx=0, rely=0.049, relwidth=1, relheight=0.951, anchor="nw")

        # Create the ImageSelectorFrame and pack it
        path_to_images = os.path.join(os.getcwd(), 'Images/Baso')
        image_selector = ImageSelectorFrame(self.page_Annotate, path_to_images)
        image_selector.place(relx=0, rely=0, relwidth=1, relheight=1, anchor="nw")


    def show_frame(self, frame,  button_text=None):
        # Bring the selected frame to the front
        frame.tkraise()

        # Highlight the selected button if button_text is provided
        if button_text:
            if self.active_button:
                # Reset the style of the previously active button
                self.active_button.configure(
                    fg_color="#2980b9",
                    text_color="white",
                    font=("Arial", 14)  # Default size
                )

            # Update the style of the newly active button
            self.active_button = self.ribbon_buttons[button_text]
            self.active_button.configure(
                fg_color="#5DADE2",
                text_color="#FFFFFF",  # Highlight text
                font=("Arial", 16, "bold")
            )


class choisirDossier(ctk.CTkFrame):
    def __init__(self, master, color, corner_radius=13, textLabel=None, textBouton=None, bouton_color=None, command=None):
        super().__init__(master=master, corner_radius=corner_radius)
        self.configure(fg_color=color, corner_radius=corner_radius)

        self.label2 = Label(self, text=textLabel)
        self.label2.place(relx=0.5, rely=0.1, anchor="n")

        self.boutonDossier = ctk.CTkButton(self, text=textBouton, command=command, fg_color=bouton_color)
        self.boutonDossier.place(relx=0.5, rely=0.5, anchor="n")  # Placed to the right of the label




class Frame(ctk.CTkFrame):
    def __init__(self, master, color, corner_radius=13):
        super().__init__(master=master, corner_radius=corner_radius)
        self.configure(fg_color=color, corner_radius=corner_radius)

class Boite(ctk.CTkEntry):
    def __init__(self, master, initial_text=""):
        super().__init__(master=master, corner_radius=10)
        self.configure(fg_color="blue", text_color="white", width=38, height=40, border_color="blue")
        self.insert(0, initial_text)
        self.bind("<KeyRelease>", self.limit_text)
        self.bind("<Return>", self.on_enter)

    def set_text(self, text):
        self.delete(0, "end")  # Clear the current text
        self.insert(0, text)  # Insert the new text

    def limit_text(self, event):
        current_text = self.get()
        if len(current_text) > 2 or not current_text.isdigit():
            self.set_text(current_text[:2])

    def on_enter(self, event):
        self.master.focus()


class Label(ctk.CTkLabel):
    def __init__(self, master, text):
        super().__init__(master=master, corner_radius=10)
        self.configure(text=text, corner_radius=13, fg_color="transparent", text_color="white")



class Button(ctk.CTkButton):
    def __init__(self, master, text, command=None):
        super().__init__(master=master, text=text, command=command)

class BouttonAccueil(ctk.CTkButton):
    def __init__(self, master, command=None):
        super().__init__(master=master, text="Accueil", command=command)
        self.configure(fg_color="blue", width=50, height=30)

class BoutonCocher(ctk.CTkFrame):
    def __init__(self, master, label_text="", command=None, variable=None, fg_color=None, hover_color=None, border_color="black"):
        super().__init__(master=master, fg_color="transparent", width=140, height=50)

        self.case = ctk.CTkCheckBox(master=self, checkbox_width=20, checkbox_height=20, variable=variable, fg_color=fg_color, text="", hover_color=hover_color, border_color=border_color)
        self.case.place(relx=0.8, rely=0.2, anchor="center")
        # Create a label and place it underneath the checkbox
        self.nomModel = ctk.CTkLabel(master=self, corner_radius=5, text_color="white", text=label_text)
        self.nomModel.place(relx=0.5, rely=0.7, anchor="center")

        # Set the command for the checkbox (if needed)
        if command:
            self.case.configure(command=command)

class StdoutRedirector:
    def __init__(self, callback=None, callback_args=None, complete_message="Visualization complete", terminal_textbox=None):
        self.callback = callback
        self.callback_args = callback_args if callback_args is not None else []
        self.complete_message = complete_message
        self.original_stdout = sys.stdout
        self.terminal_textbox = terminal_textbox  # Ajout du widget texte pour afficher stdout

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.original_stdout

    def write(self, message):
        # Output the message
        self.original_stdout.write(message)
        self.original_stdout.flush()

        # Afficher le message dans le terminal_textbox s'il est défini
        if self.terminal_textbox:
            self.terminal_textbox.configure(state="normal")  # Activer l'édition pour mettre à jour le contenu
            self.terminal_textbox.insert("end", message)  # Ajouter le message à la fin
            self.terminal_textbox.see("end")  # Faire défiler jusqu'à la fin
            self.terminal_textbox.configure(state="disabled")  # Désactiver l'édition pour éviter les modifications
            self.terminal_textbox.update()

        # Check if the complete message has been printed and call the callback with arguments
        if self.complete_message in message and self.callback:
            self.callback(*self.callback_args)  # Pass arguments to the callback function

    def flush(self):
        pass


class ImageSelectorFrame(ctk.CTkFrame):
    def __init__(self, parent, path, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.path = path

        # Configure grid layout for the frame
        self.columnconfigure(0, weight=3)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=3)

        # Create canvas to hold selectable buttons
        self.canvas = ctk.CTkCanvas(self, bg='Black')
        self.canvas.grid(row=0, column=0, sticky='nwes')

        # Create scrollbar for the canvas
        self.scroll = ctk.CTkScrollbar(self)
        self.scroll.grid(row=0, column=1, sticky='nswe')

        # Create a container frame to place into the canvas
        self.container = ctk.CTkFrame(self.canvas)
        self.container.grid(row=0, column=0, sticky='nwes')

        # Create a side container for label and delete button
        self.side_container = ctk.CTkFrame(self)
        self.side_container.grid(row=0, column=2)

        # Create a label to indicate how many images are chosen
        self.mylabel = ctk.CTkLabel(self.side_container, text='Images à supprimer: ', font=(None, 20), pady=150)
        self.mylabel.pack()

        # Create a delete button for selected images
        self.delete_button = ctk.CTkButton(self.side_container, text='Supprimer', font=(None, 20), command=self.delete)
        self.delete_button.pack()

        # Bind and configure canvas scroll region
        self.canvas.bind('<Configure>', self.update_size)
        self.canvas.after_idle(self.update_size)

        # Bind the mouse wheel to scroll the canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Dictionaries to hold image objects and their buttons
        self.image_buttons = {}

        # Get image files from the directory
        self.image_files = self.get_image_files()

        # Call the function to fill the canvas with image buttons
        self.fill_canvas()

        # List to hold selected buttons
        self.selected_buttons = []

    # Function to update canvas size
    def update_size(self, e=None):
        self.canvas["scrollregion"] = self.canvas.bbox("all")

    # Function to get image files from the directory
    def get_image_files(self):
        return [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]

    # Function to fill the canvas with image buttons
    def fill_canvas(self):

        # Get the width of the canvas
        canvas_width = self.canvas.winfo_width()

        # Set number of buttons per row
        buttons_per_row = 4
        button_size = 300

        # Calculate button width based on canvas width
        button_width = canvas_width // buttons_per_row - 10  # Subtract padding/margin

        # Adjust height if needed to maintain aspect ratio (e.g., square buttons)
        button_height = button_width

        # Clear existing widgets
        for widget in self.container.winfo_children():
            widget.destroy()


        row = 0

        # Append image objects and their names to lists
        for idx, image_file in enumerate(self.image_files):
            img = Image.open(os.path.join(self.path, image_file))
            ctk_image = ctk.CTkImage(img, size=(button_size, button_size))  # Ensure image fits button

            button = ctk.CTkButton(self.container,
                                   text=image_file,
                                   font=(None, 12),
                                   image=ctk_image,
                                   compound=ctk.TOP,
                                   fg_color='snow',
                                   command=lambda idx = idx: self.print_button(idx),
                                   width=button_width,
                                   height=button_height)

            # Calculate row and column for the button
            row = idx // buttons_per_row
            col = idx % buttons_per_row

            # Place the button in the grid
            button.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')

            self.image_buttons[idx] = button

        # Ensure the container adjusts to the buttons
        self.container.update_idletasks()
        for i in range(buttons_per_row):
            self.container.grid_columnconfigure(i, weight=1)
        self.container.grid_rowconfigure(row, weight=1)

        # Make the canvas scrollable
        self.container.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.container, anchor="nw", tags='my_tag')

        self.scroll.configure(command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set, scrollregion=self.canvas.bbox('all'))

        # Set canvas scroll regions
        def set_canvas_scrollregion(event):
            width = event.width - 4
            self.canvas.itemconfigure("my_tag", width=width)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.canvas.bind("<Configure>", set_canvas_scrollregion)

    def _on_mousewheel(self, event):
        # Scroll the canvas when the mouse wheel is used
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def print_button(self, index):
        button = self.image_buttons.get(index)
        if not button:
            return

        if index not in self.selected_buttons:
            self.selected_buttons.append(index)
            button.configure(fg_color='light blue')
        else:
            self.selected_buttons.remove(index)
            button.configure(fg_color='snow')

        self.mylabel.configure(text=f'Images à supprimer: {len(self.selected_buttons)}')

    def delete(self):
        if not self.selected_buttons:
            print('nothing is selected')
            return

        # Ensure image_files and selected_buttons are in sync
        if len(self.image_files) <= max(self.selected_buttons):
            print('Error: Selected index out of range of available images.')
            return

        for idx in sorted(self.selected_buttons, reverse=True):
            file_path = os.path.join(self.path, self.image_files[idx])
            try:
                os.remove(file_path)
                del self.image_buttons[idx]
            except FileNotFoundError:
                print(f'File not found: {file_path}')

        self.canvas.delete("all")
        self.selected_buttons = []
        self.mylabel.configure(text='Images à supprimer: 0')

        # Refresh the canvas with remaining files
        self.image_files = self.get_image_files()
        self.fill_canvas()


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        # Bind the widget to mouse enter and leave events
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        # Create a Toplevel window for the tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations (border, title bar, etc.)
        self.tooltip_window.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")  # Position near cursor

        # Break text into lines of max 80 characters
        wrapped_text = self.break_text(self.text, max_length=80)

        # Create and pack the label into the tooltip window
        label = ctk.CTkLabel(self.tooltip_window, text=wrapped_text, fg_color="lightyellow", corner_radius=6)
        label.pack(padx=5, pady=5)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()  # Close the tooltip window
            self.tooltip_window = None

    def break_text(self, text, max_length):
        lines = []
        while len(text) > max_length:
            # Find the last space within the max length to avoid cutting words
            break_point = text.rfind(' ', 0, max_length)
            if break_point == -1:  # No space found, cut at max length
                break_point = max_length

            # Append the line and trim the text
            lines.append(text[:break_point])
            text = text[break_point:].lstrip()  # Remove leading spaces for the next line

        # Append any remaining text
        if text:
            lines.append(text)

        return '\n'.join(lines)


class ImageTooltip:
    def __init__(self, widget, image_path, position="right", x_offset=10, y_offset=10):
        self.widget = widget
        self.image_path = image_path
        self.position = position
        self.tooltip_window = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.photo = None  # Keep reference to the image to manage memory properly

        # Bind the widget to mouse events
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<Button-1>", self.hide_tooltip)
        self.widget.bind("<KeyRelease>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window:
            return  # Avoid creating multiple tooltips

        # Load and resize the image for the tooltip
        image = Image.open(self.image_path)
        image = image.resize((800, 800))  # Resize as needed
        self.photo = ImageTk.PhotoImage(image)  # Store the image in the instance

        # Create a Toplevel window for the tooltip
        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations

        # Calculate the position of the tooltip
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty()

        if self.position == "right":
            x += self.widget.winfo_width()  # Right of the widget
        elif self.position == "left":
            x -= 100  # Left of the widget
        elif self.position == "top":
            y -= 100  # Above the widget
        elif self.position == "bottom":
            y += self.widget.winfo_height()  # Below the widget

        # Apply offsets and adjust vertical positioning
        x += self.x_offset
        y += self.y_offset
        y -= 550  # Adjust to position the tooltip higher

        # Set tooltip geometry
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Add the image to the tooltip
        label = ctk.CTkLabel(self.tooltip_window, image=self.photo, text="")
        label.pack(padx=5, pady=5)

        # Bind events for the tooltip window
        self.tooltip_window.bind("<Enter>", self.on_tooltip_enter)
        self.tooltip_window.bind("<Leave>", self.hide_tooltip)
        self.tooltip_window.bind("<Button-1>", self.hide_tooltip)  # Close on click
        self.tooltip_window.bind("<KeyRelease>", self.hide_tooltip)

    def hide_tooltip(self, event):
        """Destroy the tooltip window if it exists and remove the image reference."""
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass  # Prevent errors if already destroyed
            finally:
                self.tooltip_window = None
                self.photo = None  # Remove the image reference to free memory

    def on_tooltip_enter(self, event):
        """Keep the tooltip open when the mouse enters it."""
        pass


if __name__ == "__main__":
    app = Window()
    app.mainloop()