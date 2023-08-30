
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import GridSearchCV

# Function to create the model
def create_model():
    #Define the base pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(768, 768, 3))

    # Freeze the base model's layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(units=2, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#Define the learning rate schedule for Cyclical Learning Rates (CLR) to determine the appropriate learning rate for each epoch.
def clr_schedule(epoch, lr):
    if epoch % cycle_length == 0:
        return base_lr
    elif epoch % (cycle_length // 2) == 0:
        return max_lr
    else:
        return lr

#Define the Barlow Twins loss function
def barlow_twins_loss(z1, z2, lambd=0.005):
    c = tf.matmul(tf.transpose(z1), z2) / batch_size
    c = tf.reduce_mean(c, axis=1) - 1  # Remove diagonal elements
    loss = tf.reduce_sum(tf.square(c)) * lambd
    return loss

#Data augmentation            #To improve performance..Not sure about this, but the code runs smooth
image_augmentation = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.8, 1.2]
)
image_augmentation.rescale = 1.0 / 255.0  #Pixel values are rescaled to the range [0, 1]

#Load and augment the training dataset
train_generator = image_augmentation.flow_from_directory(
    '/Users/dr.elsherif/Downloads/lung_colon_image_set/colon_image_sets',
    target_size=(768, 768),   #ResNet uses 224x224 but our colon set has 768X768 pixel.
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

def augmentation_strategy(x):   #Generates two augmented views of the same image accordning to Barlow Twins method
    x1 = image_augmentation.flow(x, batch_size=1, shuffle=False).next()
    x2 = image_augmentation.flow(x, batch_size=1, shuffle=False).next()
    return [x1, x2], [x, x]

#Define the KerasClassifier with the create_model function
model = KerasClassifier(build_fn=create_model)

#Define the grid search parameters range
param_grid = {
    'batch_size': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    'epochs': [50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 100000]
}

#Define the early stopping callback   #monitors the validation loss and stops training if there is no improvement for a certain number of epochs.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Define the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(clr_schedule)

#Grid Search with Cross Validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')

#Fit the model using Grid Search and Early Stopping
grid_search.fit(train_generator, callbacks=[early_stopping, lr_scheduler])

#Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)





