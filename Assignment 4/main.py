from model import Model
from utility import read, save, draw_graphs, visualize


def run(activation_fun, layer_dims):
    global best_model
    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            print(f"\n{len(layer_dims) - 2}-Layer Model is being trained")
            print(f"Batch size: {batch_size}")
            print(f"Activation function: {activation_fun}")
            print(f"Learning rate: {learning_rate}")
            print(f"Layer dims: {layer_dims}\n")

            model = Model(layer_dims=layer_dims, learning_rate=learning_rate, num_epoch=EPOCH_SIZE,
                          batch_size=batch_size, layer_activation=activation_fun, last_layer_act_func="softmax")
            model.train(train_X, train_Y, valid_X, valid_Y)

            if best_model[-1] < model.avg_acc:
                del best_model
                best_model = (model, batch_size, activation_fun, learning_rate, layer_dims, model.avg_acc)
            else:
                del model

    print(f"\nTesting with the best hyper-parameters on the model with {len(layer_dims) - 2} hidden layers")
    best_model = best_model[0]
    save(best_model, name=f"{len(layer_dims) - 2}-{activation_fun}-layer-model")
    best_model.predict(test_X, test_Y)
    best_models.append(best_model)
    best_model = (Model, 0, "", 0.0, [], 0)


BATCH_SIZES = [16, 32, 64, 128]
DECAY_RATE = 0.998
EPOCH_SIZE = 10
LEARNING_RATES = [5e-3, 1e-2, 2e-2]
NUM_OF_CLASSES = 15
LAYER_DIMS = [120 * 120, 15]

train_X, train_Y = read(data_set_folder="Vegetable Images", mode="train")
valid_X, valid_Y = read(data_set_folder="Vegetable Images", mode="validation")
test_X, test_Y = read(data_set_folder="Vegetable Images", mode="test")

best_model = (Model, 0, "", 0.0, [], 0)
best_models = []

# Running 0-Layer NN
activation_func = "sigmoid"
run(activation_func, LAYER_DIMS)

activation_func = "relu"
run(activation_func, LAYER_DIMS)

activation_func = "tanh"
run(activation_func, LAYER_DIMS)

# Running 1-Layer NN
LAYER_DIMS = [120 * 120, 256, 15]

activation_func = "sigmoid"
run(activation_func, LAYER_DIMS)

activation_func = "relu"
run(activation_func, LAYER_DIMS)

activation_func = "tanh"
run(activation_func, LAYER_DIMS)

# Running 2-Layer NN
LAYER_DIMS = [120 * 120, 256, 128, 15]

activation_func = "sigmoid"
run(activation_func, LAYER_DIMS)

activation_func = "relu"
run(activation_func, LAYER_DIMS)

activation_func = "tanh"
run(activation_func, LAYER_DIMS)

# Saving the best model
max_acc = 0
best_model = None
for model in best_models:
    if max_acc < model.avg_acc:
        best_model = model

save(best_model, name="best-model")

best_model.display()
best_model.num_epoch = 100
best_model.init_params()
accuracy_list, loss_list = best_model.train(train_X, train_Y, valid_X, valid_Y)
# Testing the best model after 100 epoch with test data
predictions = best_model.predict(test_X, test_Y)

# Saving the best model after 100 epoch in case we need it back
save(best_model, name="best-model-100-epoch")

# Drawing the graphs for the best model with 100 epochs
draw_graphs(accuracy_list, loss_list)
visualize(best_model.params["W1"], 16, 16)
visualize(best_model.params["W2"], 8, 8)