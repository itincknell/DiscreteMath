
import math

debug_flag = True

def debug_print(message):
	if debug_flag:
		print(message)

def fancy_print(message):
	print(f"\n{message}\n" + len(message) * "=")

# Test if a number is prime
# # # # # # # # # # # # # # 

def primeTest(n,primes):
	for i in primes:
		if i > math.sqrt(n):
			break
		if n % i == 0:
			return False
	return True

# Prime Find: Find all primes up to n
# # # # # # # # # # # # # # # # # # # 

def primeFind(n):
	primes = []
	for i in range(2,n+1):
		if primeTest(i,primes):
			primes.append(i)

	if False: # Show Differences
		diffs = [int(primes[i+1]) - int(primes[i]) for i in range(len(primes)-1)]
		for i in range(len(diffs)):
			print(("Primes: " + str(primes[i + 1]) + " – " + str(primes[i] )))
			print("Difference: " + str(diffs[i]) + "\n")
	return primes

# Prime Factor utility functions

# test if factors produce the correct product
def test_factors(factors,test):
	if math.prod(factors) != test:
		print(f"Failure: math.prod(factors) != test: != {test:_}")
		exit()
	
	debug_print(f"Success: math.prod(factors) == {test:_}")
	debug_print(f"Factors: {', '.join([str(f) for f in factors])}")	

# combine repeated factors into perfect squares
def combine_duplicates(factors):	
	while( not all( factors[i] not in factors[i + 1:] for i in range(len(factors)) ) ): 
		for i in range(len(factors) - 2, 0 , -1):
			if factors[i] == factors[i + 1]:
				factors[i] *= factors[i + 1]
				factors.pop(i + 1)
		factors.sort()
	return factors

# Prime Factorization
# # # # # # # # # # # 

def prime_factor(n,primes,prime_squares=False):

	while(max([p for p in primes]) < math.sqrt(n)):
		print("\n\n\n\t\tHad to get more primes :(\n\n\n")
		primes = list(set(primes) | set(primeFind(max([p for p in primes]) * 2 )))

	debug_print(f"Factorization of {n:_}:")

	# store original value for validation
	test = n

	factors = [1]
	divides = False

	while(n > 1):

		for p in primes:
			# new prime factor
			if n % p == 0:
				divides = True
				break
			# stop running algorithm
			if p**2 > n:
				break
		
		# store factor p or n if no divisor was found
		if divides:
			x = p
		else:
			x = n
			
		# reset flag
		divides = False

		# maintains ordered list
		i = max([i for i in range(len(factors)) if factors[i] < x ]) + 1
		factors.insert(i, x)

		n //= x

	test_factors(factors,test)

	if prime_squares:
		return combine_duplicates(factors)
	else:
		return factors

# Factorize one number
# # # # # # # # # # # 

def factorize(n,primes=[]):
	if not primes:
		primes = primeFind(math.floor(math.sqrt(n)) + 100) # This margin is an educated guess
	return prime_factor(n,primes)

# Factorize a list of numbers
# # # # # # # # # # # # # # # 

def factorize_list(n_list):
	primes = primeFind(math.floor(math.sqrt(max(n_list))) + 100)
	return {n:factorize(n,primes) for n in n_list}

# fibonacci numbers
# # # # # # # # # #

def fibonacci(n):
	if n == 1:
		return [0]
	if n == 2:
		return [0,1]

	sequence = [0,1]

	for i in range(2,n):
		sequence.append(sequence[i - 1] + sequence[i - 2])

	return sequence

# extended euclidean algorithm
# # # # # # # # # # # # # # # 

def swap(a,b):
	return (a,b) if a > b else (b,a)

def extended_euclidean(a, b, info_print=True):
	
	a,b = swap(a,b)
	
	R_vector = [a,b]
	Q_vector = [0]
	X_vector = [1,0]
	Y_vector = [0,1]

	while(R_vector[-2] % R_vector[-1] != 0):
		Q_vector.append(R_vector[-2] // R_vector[-1])
		R_vector.append(R_vector[-2] %  R_vector[-1])
		X_vector.append(X_vector[-2] - (Q_vector[-1] * X_vector[-1]))
		Y_vector.append(Y_vector[-2] - (Q_vector[-1] * Y_vector[-1]))

	return R_vector[-1], X_vector[-1], Y_vector[-1]

# chinese remainder theorem
# # # # # # # # # # # # # # 

def CRT_pair(remainder, modulus, info_print=True):
	gcd, x, y = extended_euclidean(modulus[0],modulus[1],info_print=False)
	
	if modulus[0] > modulus[1]:
		temp = y
		y = x
		x = temp
	
	if gcd != 1:
		print("Moduli are not relatively prime!")
		exit()

	m = (modulus[0] * modulus[1])
	c = (x * modulus[1] * remainder[0] + y * modulus[0] * remainder[1]) % m
	
	if info_print:
		print(f"If 		x ≡ {remainder[0]} mod {modulus[0]}")
		print(f"And 	x ≡ {remainder[1]} mod {modulus[1]}")
		print(f"Then 	x ≡ {c} mod {m}")

	error = False
	if c % modulus[0] != remainder[0]:
		print(f"Error: {c} % {modulus[0]} == {c % modulus[0]} != {remainder[0]}")
		error = True
	if c % modulus[1] != remainder[1]:
		print(f"Error: {c} % {modulus[1]} == {c % modulus[1]} != {remainder[1]}")
		error = True
	if error:
		exit()

	return (c, m)

def CRT_full(remainders, moduli, info_print=True):
	if len(remainders) != len(moduli):
		print("Mismatch in number of arguments")
		return

	c,m = CRT_pair((remainders[0],remainders[1]),(moduli[0],moduli[1]),info_print)

	for i in range(2,len(moduli)):
		c,m = CRT_pair((c,remainders[i]),(m,moduli[i]),info_print)

	return c

# Uniform Cost Search and A* Search
# # # # # # # # # # # # # # # # # # 

UNEXPLORED = True
EXPLORED = False

inf = float('inf')

# distance matrix should be a square matrix with each row representing a node
# and each column representing the distances to the other reachable nodes
# infinity should be used to represent unreachable nodes for a given row

def dijkstra(distance_matrix,start_node,goal_node):

	fancy_print("Dijkstra's Algorithm / Uniform Cost Search")

	# create D shortest path array 
	D = []
	# create B unexplored boolean array
	B = []

	for i in range(len(distance_matrix)):
		D.append(inf)
		B.append(UNEXPLORED)

	D[start_node] = 0

	counter = 1

	while True:

		# find the unexplored x with the smallest value D[x]
		x = find_smallest_unexplored_node(D,B,counter)

		if x is False:
			debug_print(f"All nodes have been explored\n")
			break

		debug_print(f"The smallest unexplored node is {x}")
		debug_print(f"D[{x}] == {D[x]}")

		if x == goal_node:
			debug_print(f"Goal node reached\n")
			break

		# for every edge out of x to the node y compute D[x] + Cost[x,y]
		for y in range(len(distance_matrix[x])):

			if D[x] + distance_matrix[x][y] < D[y]:
				debug_print(f"Updating D[{y}] from {D[y]} to {D[x] + distance_matrix[x][y]}")
				D[y] = D[x] + distance_matrix[x][y]

		B[x] = EXPLORED

		counter += 1

	print(D)
	print(f"Shortest possible path from node {start_node} to {goal_node} has length {D[goal_node]}")


# Return row index of unexplored node with smallest D value

def find_smallest_unexplored_node(D,B,counter):

	debug_print(f"\nRound {counter}:")
	x = 0
	for i in range(len(D)):

		if B[x] is EXPLORED or (D[i] < D[x] and B[i] is UNEXPLORED):
			x = i

	# if the final x is already explored
	if B[x] == EXPLORED:
		return False

	return x

# distance must be an n x n matrix, and heuristic an array of length n

def a_star_search(distance_matrix,heuristic_array,start_node,goal_node):

	fancy_print("A* Search")

	# create D shortest path array 
	D = []
	# create H shortest path + hueristic value array
	H = []
	# create B unexplored boolean array
	B = []

	for i in range(len(distance_matrix)):
		D.append(inf)
		H.append(inf)
		B.append(UNEXPLORED)

	D[start_node] = H[start_node] = 0

	counter = 1

	while True:

		# find the unexplored x with the smallest value D[x]
		x = find_smallest_unexplored_node(H,B,counter)

		if x is False:
			debug_print(f"All nodes have been explored\n")
			break

		debug_print(f"The smallest unexplored node is {x}")
		debug_print(f"D[{x}] == {D[x]}")

		if x == goal_node:
			debug_print(f"Goal node reached\n")
			break

		# for every edge out of x to the node y compute D[x] + Cost[x,y]
		for y in range(len(distance_matrix[x])):

			if D[x] + distance_matrix[x][y] + heuristic_array[y] < D[y]:
				debug_print(f"Updating H[{y}] from {H[y]} to {D[x] + distance_matrix[x][y] + heuristic_array[y]}")
				D[y] = D[x] + distance_matrix[x][y]
				H[y] = D[x] + distance_matrix[x][y] + heuristic_array[y]

		B[x] = EXPLORED

		counter += 1

	print(D)
	print(f"Shortest possible path from node {start_node} to {goal_node} has length {D[goal_node]}")

