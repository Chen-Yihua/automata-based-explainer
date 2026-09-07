
# Automata Explanation Tool

This tool provides explanations for why a word is **accepted** or **rejected** by a finite automaton defined via a regular expression (in libmata or regex format).

To run this project, install the required packages listed in requirements.txt:

```pip install -r requirements.txt```

Alternatively, you can install the main dependencies individually:

```
pip install click 
pip install libmata 
pip install python-sat
```


## Usage

Run the tool from the command line:

```bash
python main.py --exp <EXPRESSION> --regex --word <WORD> [OPTIONS]

#Help
python main.py --help

#Execution of the example in the paper

python main.py -exp="(abcd+)|(ab[c-z]e+)|(bc+da)|(bc+)" --regex --word accc

# Execution of Finite Automaton Corpus with accepted word of line two (the easiest one). change '2p' by desire line
python main.py -exp="$(head -n 1 language/corpus_fa_accepted/l20m10d10/l20m10d10.txt)" --regex -w "$(sed -n '2p' language/corpus_fa_accepted/l20m10d10/l20m10d10.txt)"

# Execution of Finite Automaton Corpus with accepted word. Similar to previous case. These cases have a large enumeration. It is recommended to set timed_out and/or print explanation as they are generated as follows

python main.py -exp="$(head -n 1 language/corpus_fa_rejected/l20m10d10/l20m10d10.txt)" --regex -w "$(sed -n '2p' language/corpus_fa_rejected/l20m10d10/l20m10d10.txt)" --print_exp --timed_out 5


# DNA problem
# DNA problem does not use the complete alphaabet, for this reason we need to add the parameter --drop_alphabet [97,99,103,116] because the alphabet is a, c, g, t
python main.py -exp="$(head -n 1 language/dna/dna_810.txt)" --regex  -w "$(sed -n '2p' language/dna/dna_810.txt)" --print_exp --drop_alphabet [97,99,103,116] --target_axp --timed_out 5


# Deep Packet Inspection
# dpi_explain.py is a script that performs the union of multiple regular expressions and explains a set of input words
# To run the deep packet inspection experiment, execute:

python dpi_explain.py

# Appendix experiment - Maze
# There are 861 different mazes
# This script receives two arguments. 
# First argument represents the size of the maze, 0 is maze [10,10], 1 is maze [10,11], 2 is maze [10,12], ..., 860 is [50,50]
# The second argument represent the language, 0 is language with bounds l=1 u=1, 1 is language with bounds l=1 u=\\infty, 2 is language with bounds l=0 u=\\infty

# To run a maze experiment of size [10,10] with language l=1 u=1, execute:
python solve_maze.py 0 0

# Each one returns 15 instances because each maze is teste with 5 different words and 3 approaches. Targeting CXp, Targeting AXp, and Targeting AXp warming singleton CXps