import dlib

# Check if the svm_multiclass_linear_trainer is available
try:
    trainer = dlib.svm_multiclass_linear_trainer()
    print("svm_multiclass_linear_trainer is available.")
except AttributeError as e:
    print("Error:", e)
