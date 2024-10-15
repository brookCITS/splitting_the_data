'''
The validation_split variable specifies the proportion of the original training set that will serve as the validation set.
The original training set contains 17,000 examples. Therefore, a validation_split of 0.2 means that:

17,000 * 0.2 ~= 3,400 examples will become the validation set.
17,000 * 0.8 ~= 13,600 examples will become the new training set.

Experiment with two or three different values of validation_split. Do different values of validation_split fix the problem?

'''
from ..utils import model #build_model, train_model, plot_the_loss_curve

# The following variables are the hyperparameters.
learning_rate = 0.08
my_epochs = 30
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.3

# Identify the feature and the label.
my_feature = "median_income"    # the median income on a specific city block.
my_label = "median_house_value" # the median house value on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on the neighborhood's median income.



def run(context):
    # Invoke the functions to build and train the model.
    train_df = context.train_df
    my_model = model.build_model(context,learning_rate)
    epochs, rmse, history = model.train_model(context, my_model, train_df, my_feature,
                                        my_label, my_epochs, batch_size,
                                        validation_split)

    model.plot_the_loss_curve(context, epochs, history["root_mean_squared_error"],
                        history["val_root_mean_squared_error"])
def test(): 
    question = "Do different values of validation_split fix the problem?"
    answer = "Not really, the error difference is huge no matter the validation spllit"

    return {"question":question,"answer": answer}