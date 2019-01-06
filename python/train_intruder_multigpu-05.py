from sklearn.datasets import load_files
from keras.utils import np_utils

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras_sequential_ascii import sequential_model_to_ascii_printout
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from models_intruderNet import ModelIntruderNet38 as model_intruder
import time
from keras.preprocessing import image
from tqdm import tqdm
from keras.utils import multi_gpu_model

start_time = time.time()

epochs = model_intruder().get_epochs()
save_name = model_intruder().get_modelname()
num_classes = 2


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), 2)
    return image_files, image_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('./images/train')
valid_files, valid_targets = load_dataset('./images/valid')
test_files, test_targets = load_dataset('./images/test')
'''
#print("train files: ", train_files)
#print("train targets: ", train_targets)
print("Train 0: ", train_targets[0], train_files[0])
print("Train 1: ", train_targets[1], train_files[1])
print("Train 10: ", train_targets[10], train_files[10])
#print("valid files: ", valid_files)
#print("valid targets: ", valid_targets)
#print("test files: ", test_files)
#print("test targets: ", test_targets)
print("test 0: ", test_targets[0], test_files[0])
print("test 1: ", test_targets[1], test_files[1])
print("test 10: ", test_targets[10], test_files[10])

print("valid 0: ", valid_targets[0], valid_files[0])
print("valid 1: ", valid_targets[1], valid_files[1])
print("valid 10: ", valid_targets[10], valid_files[10])
'''

# load list of ipcam names
intruder_names = [item[15:-1] for item in sorted(glob("./images/train/*/"))]

# print statistics about the dataset
print('There are %d total intruder categories.' % len(intruder_names))
print("Intruder categories: ", intruder_names)
print('There are %s total intruder images.' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training intruder images.' % len(train_files))
print('There are %d validation intruder images.' % len(valid_files))
print('There are %d test intruder images.' % len(test_files))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(640, 480))
    x = image.img_to_array(img)
    #print(img_path, img, x)
    #plt.imshow(img)
    #plt.show()
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

	
#pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

print("train_tensors size: ", len(train_tensors))
print("train_tensors: ", train_tensors.shape)

model = model_intruder().get_model_compiled_multi_gpu()
model.summary()

check_pointer = ModelCheckpoint(filepath="saved_models/"+save_name+".hdf5",
                               verbose=1, save_best_only=True)

model_time = time.time()
print("Time elapsed: ", (model_time - start_time))

cnn = model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[check_pointer], verbose=1)

model.load_weights("saved_models/"+save_name+".hdf5")

end_time = time.time()
print("time elapsed total, data, epoch: ", (end_time - start_time), (model_time - start_time), (end_time - model_time))

# get index of predicted intruder categories for each image in test set
intruder_category_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
print("")


# report test accuracy
test_accuracy = 100*np.sum(np.array(intruder_category_predictions)==np.argmax(test_targets, axis=1))/len(intruder_category_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


sequential_model_to_ascii_printout(model)

#https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/

#plt.figure(0)
plt.plot(cnn.history['acc'], 'r')
plt.plot(cnn.history['val_acc'], 'g')
plt.xticks(np.arange(0, (epochs + 2), 10.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])
plt.savefig("saved_results/"+save_name+"_acc.jpg", dpi=150)
plt.close()

#plt.figure(1)
plt.plot(cnn.history['loss'], 'r')
plt.plot(cnn.history['val_loss'], 'g')
plt.xticks(np.arange(0, (epochs + 2), 10.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])
plt.savefig("saved_results/"+save_name+"_loss.jpg", dpi=150)
#plt.show()

target_names = ['Empty', 'Intruder']
#print("intruder_category_predictions:", intruder_category_predictions)
#print("test_targets: ", test_targets)
#print("test_targets: ", np.argmax(test_targets, axis=1))


print("")
print("Confusion Matrix")
print(confusion_matrix(np.argmax(test_targets, axis=1), intruder_category_predictions))

print("")
print("Classification Report")
print(classification_report(np.argmax(test_targets, axis=1), intruder_category_predictions, target_names=target_names))

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
print("")
print("ROC - Receiver Operating Characteristic")
fpr_false_postive_rates, tpr_true_positive_rates, thresholds = roc_curve(np.argmax(test_targets, axis=1), intruder_category_predictions)
print("fpr, tpr, threshold:", fpr_false_postive_rates, tpr_true_positive_rates, thresholds)

print("")
print("AUC - Area Under Curve")
auc_val = auc(fpr_false_postive_rates, tpr_true_positive_rates)
print("auc: ", auc_val)

end_time = time.time()
total_time = end_time - start_time
print("total, start, end: ", total_time, start_time, end_time)

