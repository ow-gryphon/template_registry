# Test Directory

Testing is a core part of software engineering and data analysis. Python provides many utilities 
for automating the testing process to ensure that the functions we write in our core analysis and
 codebase do exactly what we expect them to do and that new changes do not alter existing behavior.
 
This folder should be used to expand and build automated unit tests. Specifically we use a 
library called [pytest](https://docs.pytest.org/en/latest/) to help structure and execute tests. 

To run the full, automated test suite, navigate up to the the root directory of this repository 
and run `pytest`. This will kick off the automatic execution of all the unit tests. 

Scripts, or files with `.py` that begin with the word "test" will be inspected, and any 
functions or classes that begin with the word 'test' will be executed.
