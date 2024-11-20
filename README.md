# learning_maximum_matching

REPO for the Reinforcement Learning and Algorithm Discovery seminar by the cutting-edge research group "AI & Sustainability" at HPI 

## Abstract Intruction Set
The instruction set could contain the following operations:
- A set of instructions to manipulate program flow (this includes an instruction to add
a marker in the program, as well as an instruction to jump to the last added point
and remove it, as well as an instruction to remove the last added jump point without
changing flow)
- A set of instructions to manage and stack pointers in the graph (Add a new pointer at
the position of the current one, move the current pointer to adjacent nodes or edges,
remove the current pointer)
- A set of instructions to read/write data to short-term memory.
- Two commands to compare the values around the current pointer and the short-term
memory. If the result is positive, the following command is executed, otherwise it is
skipped. Those operations might include fundamental arithmetic to allow aggregation
of values.

