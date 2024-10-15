'''
The test set usually acts as the ultimate judge of a model's quality. 
The test set can serve as an impartial judge 
because its examples haven't been used in training the model. 

Look at the test results from task 3

Compare the root mean squared error of the model when evaluated on each of the three datasets:

    training set: look for root_mean_squared_error in the final training epoch.
    validation set: look for val_root_mean_squared_error in the final training epoch.
    test set: run the preceding code cell and examine the root_mean_squared_error.


'''

def test(): 
    
    question = "Ideally, the root mean squared error of all three sets should be similar. Are they?"
    answer = "yes they are similar"

    return {"question":question,"answer": answer}