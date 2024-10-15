'''
To fix the problem, shuffle the examples in the training set
before splitting the examples into a training set and validation set. 
To do so, take the following steps

1) Shuffle the data in the training set by using train_df.reindex( ) function || train_df.reindex(np.random.permutation(train_df.index))
2) Pass shuffled_train_df (instead of train_df) as the second argument to train_model( ) function

The test set usually acts as the ultimate judge of a model's quality. 
The test set can serve as an impartial judge 
because its examples haven't been used in training the model. 

3) Evaluate the model on the test set after training it.
'''
from ..utils.model import build_model, train_model, plot_the_loss_curve

# The following variables are the hyperparameters.
learning_rate = 0.08
my_epochs = 70
batch_size = 100

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.2

# Identify the feature and the label.
my_feature = "median_income"    # the median income on a specific city block.
my_label = "median_house_value" # the median house value on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on the neighborhood's median income.



# Shuffle the examples.
shuffled_train_df = '[code goes here]'

def run(context):
    # Invoke the functions to build and train the model.
    train_df = context.train_df
    my_model = build_model(context,learning_rate)
    epochs, rmse, history = train_model(context, my_model, shuffled_train_df, my_feature,
                                        my_label, my_epochs, batch_size,
                                        validation_split)

    plot_the_loss_curve(context, epochs, history["root_mean_squared_error"],
                        history["val_root_mean_squared_error"])
    
    

    test_df = context.test_df
    
    x_test = test_df[my_feature]
    y_test = test_df[my_label]

    results = my_model.evaluate(x_test, y_test, batch_size=batch_size)
    print(results)

    

    question = "At what range of values of validation_split do the final loss values for the training set and validation set diverge meaningfully? Why?"
    answer = ""


    return {"question":question,"answer": answer}

