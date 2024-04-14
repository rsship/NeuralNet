run:
	# clang -Wall -Wextra -o main main.c -lm
	clang -Wall -Wextra -o nn nn.c -lm
prod: 
	clang -03 -Wall -Wextra -o nn nn.c -lm
.PHONY: run prod
