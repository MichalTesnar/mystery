from imports import *
from constants import *

# Approximated function

def unpack_int(b):
    while not isinstance(b, int) and not isinstance(b, np.float64) and not isinstance(b, np.float32):
        b = b[0] 
    return b


def toy_function(input):
    output = []
    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)
        out = math.sin(inp)  # + np.random.normal(0, std)
        output.append(10 * out)
    return np.array(output)

def get_data():
    x_train = np.linspace(-4.0, 4.0, num=SAMPLE_RATE)
    train_indices = np.arange(SAMPLE_RATE)
    # np.random.shuffle(train_indices)
    x_train = x_train[train_indices]
    y_train = toy_function(x_train)
    domain = np.linspace(-4.0, 4.0, num=100)
    domain = domain.reshape((-1, 1))
    domain_y = toy_function(domain)
    return x_train, y_train, domain, domain_y


def get_model(name):
    if name == "Ensembles":
        return get_ensembles()
    if name == "Dropout":
        return get_dropout()
    if name == "GRP":
        return get_grp()
    raise Exception("UTILS: Invalid model name / model not implemented.")


def retrain_model(name, current_model, current_x_train, current_y_train, extra_epochs=0, batch_size=32):
    if name == "Ensembles":
        return retrain_ensembles(current_model, current_x_train, current_y_train, extra_epochs, batch_size=batch_size)
    if name == "Dropout":
        return retrain_dropout(current_model, current_x_train, current_y_train, extra_epochs, batch_size=batch_size)
    if name == "GRP":
        return retrain_grp(current_model, current_x_train, current_y_train)
    raise Exception("UTILS: Invalid model name / model not implemented.")


def pred_model(name, current_model, domain):
    if name == "Ensembles":
        return pred_ensembles(current_model, domain)
    if name == "Dropout":
        return pred_dropout(current_model, domain)
    if name == "GRP":
        return pred_grp(current_model, domain)
    raise Exception("UTILS: Invalid model name / model not implemented.")


################################## ENSEMBLES ##############################################
# model definition
def get_ensembles():
    SIMPLE = False
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(128, activation="relu")(inp)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")
        if SIMPLE:
            return train_model
        return train_model , pred_model

    if SIMPLE:
        model = SimpleEnsemble(model_fn, num_estimators=10)
    else:
        model = DeepEnsembleRegressor(model_fn, num_estimators=10)
    return model


def pred_ensembles(model, domain):
    pred_mean, pred_std = model.predict(domain)
    return pred_mean, pred_std


def retrain_ensembles(model, x_train, y_train, extra_epochs, batch_size=32):
    model.fit(x_train, y_train, verbose=False, epochs=EPOCHS + extra_epochs, batch_size=batch_size)
    return model

################################## DROPOUT ##############################################
# model definition
def get_dropout(prob=0.2):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(1,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(128, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(128, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(64, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def retrain_dropout(model, x_train, y_train, extra_epochs, batch_size=32):
    model.fit(x_train, y_train, verbose=False, epochs=EPOCHS+extra_epochs, batch_size=batch_size)
    return model


def pred_dropout(model, point):
    mc_model = StochasticRegressor(model)
    pred_mean, pred_std = mc_model.predict(point, num_samples=NUM_SAMPLES)
    return pred_mean, pred_std


################################## Gaussian Process Regression ################################
def get_grp():
    
    kernel = gp.kernels.ConstantKernel(5, (1e-3, 1e3)) * gp.kernels.RBF(5, (1e-3, 1e3))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    return model

def retrain_grp(model, x_train, y_train):
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def pred_grp(model, point):
    pred_mean, pred_std = model.predict(point, return_std=True)
    return pred_mean, pred_std