"""

A region proposal network for localising tissue
sections in whole slide images.

"""



from od_utils import *

from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D
from keras.models import Model
from keras.optimizers import Adam


# Import model to get weights...
mn = MobileNet(input_shape=(224, 224, 3), include_top=False)
weights = mn.get_weights()
del mn

# Start fresh with custom input and load weights
K.clear_session()
base = MobileNet(input_shape=(None, None, 3), include_top=False, weights=None)
base.set_weights(weights)


# Base Network
input = base.input
x = base.layers[-1].output

# Region Proposal Network
# 3 sizes - 64x64, 128x128 256x256 with aspect ratio - 1:1, 2:1, 1:2
num_anchors = 9
x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(x)
x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
x_regr = Conv2D(num_anchors*4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

# Build it...
model = Model(inputs=[input], outputs=[x_class, x_regr])


# Lock weights in CNN for train
for layer in model.layers[0:-4]:
    layer.trainable = False

optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])



# --------------------------------- Train --------------------------------- #
annotation_dir = "/home/simon/PycharmProjects/ObjectDetection/data/annotations/"
image_dir = "/home/simon/PycharmProjects/ObjectDetection/data/images/"

data_gen = DataGenerator(annotation_dir, image_dir)


training_steps = 1000
print(model.metrics_names)
for step in range(training_steps):

    X, Y_object, Y_regression = next(data_gen)

    history = model.train_on_batch(X, [Y_object, Y_regression])
    print(step, history)
    #if history[0] < 1:
    #    print("Stopping early at step", step)
    #    break


#model.save_weights("./weights/object_detect_steps_{0:04}.h5".format(step+1))


# Load weights
#model.load_weights("./weights/object_detect_steps_0056.h5")


X, Y_object, Y_regression = next(data_gen)
objects, offsets = model.predict(X)

plt.imshow(X[0])
ax = plt.gca()

for r in range(objects[0].shape[0]):
    for c in range(objects[0].shape[1]):
        anchor = np.argmax(objects[0][r, c], axis=-1)

        # Get anchor values
        width, height = data_gen.anchors[anchor]

        x, y = data_gen.transform_small_coords(r, c)

        # Apply offsets
        x, y, width, height = apply_offest(x, y, width, height, offsets[0][r, c][4*anchor:(4*anchor) + 4])

        # Get class prediction
        prob = objects[0][r, c, anchor]

        if prob < 0.5:
            continue

        if x + width > data_gen.im_cols or y + height > data_gen.im_rows:
            continue

        print("Probability:", prob)

        bbox = BBox((x, y), width, height, linewidth=2, edgecolor=green, facecolor='none')
        ax.add_patch(bbox)

        # Apply offsets
        #

plt.show()













