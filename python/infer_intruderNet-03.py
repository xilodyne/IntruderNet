from sklearn.datasets import load_files
from keras.models import Sequential, load_model
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import h5py
from models_intruderNet import ModelIntruderNet38 as model_intruder
from keras.utils import multi_gpu_model


debug = True
#save_name = "mod38_parallel_2.i10"
save_name = model_intruder().get_modelname()
model = load_model("saved_models/"+save_name+".hdf5")


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    img_files = np.array(data['filenames'])
    if debug:
        print(path)
        print(data['filenames'])
        #print(img_files)
    return img_files


def path_to_tensor(get_images_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(get_images_path, target_size=(640, 480))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    #print(img_path, img, x)
    #plt.imshow(img)
    #plt.show()
    return np.expand_dims(x, axis=0)


def paths_to_tensor(get_images_path):
    list_of_tensors = [path_to_tensor(get_images_path) for get_images_path in tqdm(get_images_path)]
    return np.vstack(list_of_tensors)


def get_inferences_by_category(get_images_path):
    image_files = load_dataset(get_images_path)
    # pre-process the data for Keras
    image_tensors = paths_to_tensor(image_files).astype('float32')/255

    # get index of predicted intruder categories for each image in test set
    intruder_cat_pred = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in image_tensors]

    if debug:
        print(intruder_cat_pred)
    #create an array of same size full of zeros
    test_tar = np.full((1, len(intruder_cat_pred)), 0)

    return intruder_cat_pred, test_tar


def get_results(intruder_cat_pred, category):
    # only empty category should have a zero as a correct result
    if category == "empty":
        correct = intruder_cat_pred.count(0)
        incorrect = intruder_cat_pred.count(1)
    else:
        correct = intruder_cat_pred.count(1)
        incorrect = intruder_cat_pred.count(0)

    test_acc = correct / len(intruder_cat_pred)

    print('Test accuracy for %s images: %.2f%% (correct %5d, incorrect %5d)' %
          (category, test_acc, correct, incorrect))
    return


print("model: ", save_name+".hdf5")

print('')
img_category = 'empty'
img_path = './images/infer/' + img_category
intruder_category_predictions, test_targets = get_inferences_by_category(img_path)
get_results(intruder_category_predictions, img_category)

print('')
img_category = 'intruder'
img_path = './images/infer/' + img_category
intruder_category_predictions, test_targets = get_inferences_by_category(img_path)
get_results(intruder_category_predictions, img_category)

print('')
img_category = 'intruder_traces'
img_path = './images/infer/' + img_category
intruder_category_predictions, test_targets = get_inferences_by_category(img_path)
get_results(intruder_category_predictions, img_category)

print('')
img_category = 'door'
img_path = './images/infer/' + img_category
intruder_category_predictions, test_targets = get_inferences_by_category(img_path)
get_results(intruder_category_predictions, img_category)

print('')
img_category = 'night'
img_path = './images/infer/' + img_category
intruder_category_predictions, test_targets = get_inferences_by_category(img_path)
get_results(intruder_category_predictions, img_category)

