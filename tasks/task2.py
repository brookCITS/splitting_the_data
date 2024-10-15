'''
No matter how you split the training set and the validation set, 
the loss curves differ significantly. Evidently, 
the data in the training set isn't similar enough to the data in the validation set.


To determine why the loss curves aren't highly similar, 
use the pandas head(n) method and output the first n rows of the DataFrame and 
the pandas tail(n) method to output the last n rows of the DataFrame.


'''

def run(context):
    train_df = context.train_df
    # Print the first 10 rows of the training DataFrame
    print('frist 10 rows')


    # Print the last 10 rows of the training DataFrame
    print('last 10 rows')


    question = "What do you notice?"
    answer = ""

    return {"question":question,"answer": answer}