/*
	CpE 561 - Parallel Computing
	Assignment 2

	Hasan Al-Faisal - 220125849
	hasan.alfaisal@ku.edu.com

	Fall 2021 -- 19/12/2021

	Page Rank (PR) & Hybrid Breadth-First Search (BFS) 
	Sequential & Parallel Implementations
*/

// Include necessary libraries
#include <iostream>
#include <chrono>
#include <list>
#include <omp.h>
#include <stdio.h>
#include <string.h>

using namespace std;

const int MAXsize = 12; // Number of nodes in the network/graph
const int MAXlinks = 4; // Maximum number of links per node

int edges = 0; // Counter for the number of edges

// Node structure
struct node
{
	int value = 0; // Node value
	bool passed = false; // Node has been passed?	
	node* links[MAXlinks] = {}; // Array of nodes it links to
};

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------

// Print all nodes and their links
void printNodes(node root[]) 
{
	edges = 0; // Zero the counter in case we need to use the function more than once

	// Print network/graph size
	if (MAXsize < 100) cout << "Here are the randomly generated node in the network/graph:" << endl;
	else cout << "Generating nodes..." << endl;
	cout << "Total number of nodes: (" << MAXsize << "), where each noted is permitted: (";
	cout << MAXlinks << ") links" << endl;
	if (MAXsize < 100) cout << endl << "------------------------------------------------------------------" << endl;
	
	for (int i = 0; i < MAXsize; i++)
	{
		if (MAXsize < 100)		
			printf("Node: %4d  links to: ", root[i].value); // Print node value

		for (int j = 0; j < MAXlinks; j++)
		{
			if (root[i].links[j] != 0)
			{
				if (MAXsize < 100)				
					printf("%4d", root[i].links[j]->value); // Print links' value
				edges++; // Count number of edges in the network/graph
			}
			else { if (MAXsize < 100) cout << "   x"; } // Print x in place of null links

			if (MAXsize < 100) {
				if (j != MAXlinks - 1) { cout << ", "; } // Print comma between node links
				else { cout << endl; } // New line after each node
			}		
		}
	}

	cout << endl << "Number of edges is: (" << edges << ")" << endl;
	cout << "------------------------------------------------------------------" << endl << endl;
}

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------

//BFS using Queue
void BFS_Q(node* source, bool parallel)
{	
	list<node> nQueue; // Create Queue
	nQueue.push_back(*source); // Push first node to the back of the queue
	(*source).passed = true; // Mark it as passed

	// Check to execute in parallel or not
	if (parallel)
	{
		// While the Queue is not empty
		while (!nQueue.empty()) {

			node tempN = nQueue.front(); // Get the front node of the queue
			cout << tempN.value << " "; // Print it

			nQueue.pop_front(); // Pop printed node
			tempN.passed = true; // Mark it as passed

			// Check node's links
			#pragma omp parallel for
			for (int i = 0; i < MAXlinks; i++)
			{
				// If link exists and has not been passed yet
				if (tempN.links[i] && !tempN.links[i]->passed)
				{
					tempN.links[i]->passed = true; // Mark it as passed
					#pragma omp critical
						nQueue.push_back(*tempN.links[i]); // Push it into the queue						
				}
			}
		}
	}
	else
	{
		// While the Queue is not empty
		while (!nQueue.empty()) {

			node tempN = nQueue.front(); // Get the front node of the queue
			cout << tempN.value << " "; // Print it

			nQueue.pop_front(); // Pop printed node
			tempN.passed = true; // Mark it as passed

			// Check node's links
			for (int i = 0; i < MAXlinks; i++)
			{
				// If link exists and has not been passed yet
				if (tempN.links[i] && !tempN.links[i]->passed)
				{
					nQueue.push_back(*tempN.links[i]); // Push it into the queue
					tempN.links[i]->passed = true; // Mark it as passed
				}
			}
		}
	}	
}

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------

//BFS using an array
void BFS_A(node* source, bool parallel)
{
	node* nextNode[MAXsize * MAXlinks] = {};
	int Index = 0;
	int CurrIndex = 0;

	//Check source node just in case
	if (source != 0 && !source->passed)
	{
		cout << source->value << " ";
		source->passed = true;
	}
	else 
	{
		cout << endl << "------------------------------------------------------------------" << endl;
		cout << "Source node does not exist" << endl;
		cout << "------------------------------------------------------------------" << endl << endl;
		return;
	}

	// While the number of passed nodes is less than the total nodes in the network/graph
	if (parallel)
	{
		while (1)
		{
			// Print the adjacent nodes in parallel
			#pragma omp parallel for
			for (int i = 0; i < MAXlinks; i++)
			{
				// If adjacent node exists has not been printed yet
				if (source->links[i] && !source->links[i]->passed)
				{
					source->links[i]->passed = true; // Mark it as passed
					printf("%d ", source->links[i]->value);
					#pragma omp critical
					{
						nextNode[CurrIndex] = source->links[i]; // Store it in array				
						CurrIndex++; // Increment counter
					}
				}
			}

			// Move to the next node using the array
			if (nextNode[Index])
			{
				source = nextNode[Index];
				Index++;
			}
			else return;
		}
	}
	else
	{
		while (1)
		{
			// Print the adjacent nodes
			for (int i = 0; i < MAXlinks; i++)
			{
				// If adjacent node exists has not been printed yet
				if (source->links[i] && !source->links[i]->passed)
				{				
					nextNode[CurrIndex] = source->links[i]; // Store it in array
					cout << source->links[i]->value << " "; // Print it		
					source->links[i]->passed = true; // Mark it as passed
					CurrIndex++; // Increment counter
				}
			}

			// Move to the next node using the array
			if (nextNode[Index])
			{
				source = nextNode[Index];
				Index++;
			}
			else return;
		}
	}	
}

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------

// Page Rank Implementation
// Page Rank slowed down from parallelism, so parallelism's lines have been removed (turned to comments)
void PageRank(node source[MAXsize], bool parallel)
{
	// Necessary variables
	float PRmatrix[MAXsize][MAXsize] = {};
	float transition[MAXsize][MAXsize] = {};
	float current[MAXsize][MAXsize] = {};

	float d = 0.85f;
	float rowSum = 0;
	float sum = 0;
	float temp = 0;

	if (MAXsize > 15)
	{
		cout << "*Executing Page Rank algorithm..." << endl;
		cout << "*Matrices will only be displayed for networks/graphs of size 15 and less" << endl;
		if (MAXsize > 100) cout << "*Resulting Page Ranks matrix will only be displayed for networks/graphs of size 100 and less" << endl;
		cout << "*Because otherwise they may become unreadable" << endl << endl; // and can lead to stack overflow
	}

	// Turn the nodes in the network/graph into a matrix
	
	if (parallel)
	{
		#pragma omp parallel for
		for (int i = 0; i < MAXsize; i++)
		{
			for (int j = 0; j < MAXlinks; j++)
			{
				if (source[i].links[j] != 0)
				{
					PRmatrix[i][source[i].links[j]->value - 1] = 1;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < MAXsize; i++)
		{
			for (int j = 0; j < MAXlinks; j++)
			{
				if (source[i].links[j] != 0)
				{
					PRmatrix[i][source[i].links[j]->value - 1] = 1;
				}
			}
		}
	}

	// Print the resulting incidence matrix
	if (MAXsize <= 15)
	{
		// Print Matrix title
		cout << "Converting the network/graph's nodes into an incidence matrix:" << endl << endl;
		cout << "        ";
		// Display column number
		for (int i = 0; i < MAXsize; i++)
		{
			printf("   %4d ", i + 1);
		}
		cout << endl << endl;

		for (int i = 0; i < MAXsize; i++)
		{
			// Display row number
			printf("> %4d    ", i + 1);

			for (int j = 0; j < MAXsize; j++)
			{
				printf(" %1.4f ", PRmatrix[i][j]); // Display row content
			}
			cout << endl;
		}

		// Print Matrix title
		cout << endl << "------------------------------------------------------------------" << endl << endl;
		cout << "This is the corresponding transition matrix:" << endl << endl;

		// Display column number
		cout << "        ";		
		for (int i = 0; i < MAXsize; i++)
		{
			printf("   %4d ", i + 1);
		}
		cout << endl << endl;
	}
	
	// Find and Print the corresponding transition matrix
	for (int i = 0; i < MAXsize; i++)
	{
		// Display row number
		if (MAXsize <= 15) printf("> %4d    ", i + 1);
		rowSum = 0;

		// Calculate row sum
		for (int j = 0; j < MAXsize; j++)
		{
			if (PRmatrix[i][j]) rowSum++; 
		}

		if (parallel)
		{
			#pragma omp parallel for
			for (int j = 0; j < MAXsize; j++)
			{
				// If row is empty -- prevent the "Dangling Node" problem
				if (rowSum == 0) transition[i][j] = 1 / rowSum;
				else transition[i][j] = PRmatrix[i][j] / rowSum;
			}
		}
		else
		{
			for (int j = 0; j < MAXsize; j++)
			{
				// If row is empty -- prevent the "Dangling Node" problem
				if (rowSum == 0) transition[i][j] = 1 / rowSum;
				else transition[i][j] = PRmatrix[i][j] / rowSum;
			}
		}
		
		if (MAXsize <= 15)
		{
			for (int j = 0; j < MAXsize; j++)
			{
				printf(" %1.4f ", transition[i][j]); // Display row content
			}
		}	
		if (MAXsize <= 15) cout << endl;
	}

	if (MAXsize <= 15)
	{
		// Print Matrix title
		cout << endl << "------------------------------------------------------------------" << endl << endl;
		cout << "This is the modified transition matrix:" << endl << endl;

		// Apply the Page Rank formula
		cout << "        ";
		// Display column number
		for (int i = 0; i < MAXsize; i++)
		{
			printf("   %4d ", i + 1);
		}
		cout << endl << endl;
	}

	for (int i = 0; i < MAXsize; i++)
	{
		// Display row number
		if (MAXsize <= 15) printf("> %4d    ", i + 1);
		for (int j = 0; j < MAXsize; j++)
		{
			// Apply page rank formula with d = 0.85
			temp = transition[i][j];
			temp = (temp * d) + ((1-d) / MAXsize);
			transition[i][j] = temp;

			if (MAXsize <= 15) printf(" %1.4f ", transition[i][j]); // Display row content
		}
		if (MAXsize <= 15) cout << endl;
	}

	if (MAXsize <= 15) cout << endl << "------------------------------------------------------------------" << endl;

	// Calculate the Identity Matrix
	if (parallel)
	{
		#pragma omp parallel for
		for (int i = 0; i < MAXsize; i++)
		{
			for (int j = 0; j < MAXsize; j++)
			{
				current[i][j] = (i == j) ? 1.0f : 0.0f;
			}
		}
	}
	else
	{
		for (int i = 0; i < MAXsize; i++)
		{
			for (int j = 0; j < MAXsize; j++)
			{
				current[i][j] = (i == j) ? 1.0f : 0.0f;
			}
		}
	}


	// Initialize step number to represent matrix power
	int step = 0;
	do {
		// Compute the next matrix power
		float product[MAXsize][MAXsize] = {};

		// Find the product
		for (int i = 0; i < MAXsize; i++)
			for (int j = 0; j < MAXsize; j++)
			{
				double	sum = 0.0;
				for (int k = 0; k < MAXsize; k++)
					sum += current[i][k] * transition[k][j];
				product[i][j] = sum;
			}
		// Copy the product to current, to be used in the next iteration
		memcpy(current, product, sizeof(PRmatrix));

		// Check if the stationary vector has been reached
		double	diff, square_diff = 0.0;
		for (int j = 0; j < MAXsize; j++)
			for (int i = 1; i < MAXsize; i++)
			{
				diff = (current[i][j] - current[0][j]);
				square_diff += diff * diff;
			}
		if (square_diff < 0.0000000001) break;
		else ++step;
	} while (1);

	if (MAXsize <= 15)
	{
		// Print Matrix title
		printf("\nProbabilities stabilize at:");
		printf("\nThe Transition Matrix to power %d: \n", step + 1);
		cout << "        ";
		// Display column number
		for (int i = 0; i < MAXsize; i++)
		{
			printf("   %4d ", i + 1);
		}
		cout << endl << endl;

		// Display rows
		for (int i = 0; i < MAXsize; i++)
		{
			//Display row number
			printf("> %4d    ", i + 1);
			for (int j = 0; j < MAXsize; j++)
				printf(" %1.4f ", current[i][j]); // Display row content
			printf("\n");
		}
		printf("\n\n");
	}

	// Display the stabilized page ranks

	// Now that Page Rank has stabilized, it is the same for each row
	float rank[MAXsize];
	int	node[MAXsize];

	if (parallel)
	{
		#pragma omp parallel for
		for (int j = 0; j < MAXsize; j++)
		{
			node[j] = j;
			rank[j] = current[0][j];
		}
	}
	else
	{
		for (int j = 0; j < MAXsize; j++)
		{
			node[j] = j;
			rank[j] = current[0][j];
		}
	}

	// Display the Page Rank of nodes
	if (MAXsize < 100)
	{
		printf("\nHere is the resulting stationary vector: \n");
		for (int j = 0; j < MAXsize; j++)
		{
			printf(" %4d: ", node[j] + 1);
			printf("%1.4f", rank[j]);
		}
	}

	// Sort the Page Rank of nodes
	// perform a bubble-sort on the stationary vector's components
	int	i = 0, j = 1;
	do {
		if (rank[i] < rank[j])
		{
			int	node_i, node_j;
			double	temp_i, temp_j;
			temp_i = rank[i];
			temp_j = rank[j];
			rank[i] = temp_j;
			rank[j] = temp_i;
			node_i = node[i];
			node_j = node[j];
			node[i] = node_j;
			node[j] = node_i;
			i = 0;
			j = 1;
		}
		else {
			++i;
			++j;
		}
	} while (j < MAXsize);

	// Display the sorted Page Rank of nodes
	if (MAXsize < 100)
	{
		printf("\n\nHere is the resulting page-rank vector: \n");
		for (int j = 0; j < MAXsize; j++)
		{
			printf(" %4d: ", node[j] + 1);
			printf("%1.4f", rank[j]);
		}
		cout << endl;
		cout << endl << "------------------------------------------------------------------" << endl << endl;
	}

}

//------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------

int main()
{
	//------------------------------------------------------------------------------------------------------------
	// Setting up the environment
	//------------------------------------------------------------------------------------------------------------

	// Array of nodes
	struct node nodes[MAXsize];

	// First and last nodes links counters
	int FirstNodeLinksNum = 0, LastNodeLinksNum = 0;

	// Initialize array values
	for (int i = 0; i < MAXsize; i++)
		nodes[i].value = (i + 1); // nodes[i].passed = false;

	srand(time(0)); // Randomizer

	// Randomize node links
	for (int i = 0; i < MAXsize; i++)
	{
		for (int j = 0; j <= MAXlinks; j++)
		{
			int linkTo = rand() % MAXsize; // Generate random node number

			// Assign a random link to random node
			nodes[i].links[rand() % MAXlinks] = &nodes[linkTo];
		}
	}

	for (int j = 0; j <= MAXlinks; j++)
	{
		// Count non-null links in the first node
		if (nodes[0].links[j] != NULL) FirstNodeLinksNum++;
		// Count non-null links in the last node
		if (nodes[MAXsize - 1].links[j] != NULL) LastNodeLinksNum++;
	}

	// Print all nodes and their links if the number of nodes in the network/graph is less than 100
	printNodes(nodes);	

	//------------------------------------------------------------------------------------------------------------
	// Page Rank
	//------------------------------------------------------------------------------------------------------------

	auto startPR_NP = std::chrono::high_resolution_clock::now(); // Get time before PR

	// Calculate Page Rank for the nodes in the network/graph
	PageRank(nodes, false); // Call the Page Rank function

	auto endPR_NP = std::chrono::high_resolution_clock::now(); // Get time after PR

	// Calculate and display execution time
	auto executionTimePR_NP = std::chrono::duration_cast<std::chrono::microseconds>(endPR_NP - startPR_NP).count();

	//------------------------------------------------------------------------------------------------------------
	// Page Rank - parallelized -- slower than sequential
	//------------------------------------------------------------------------------------------------------------

	//auto startPR = std::chrono::high_resolution_clock::now(); // Get time before PR

	// Calculate Page Rank for the nodes in the network/graph
	//PageRank(nodes, true); // Call the Page Rank function

	//auto endPR = std::chrono::high_resolution_clock::now(); // Get time after PR

	// Calculate and display execution time
	//auto executionTimePR = std::chrono::duration_cast<std::chrono::microseconds>(endPR - startPR).count();

	//------------------------------------------------------------------------------------------------------------
	// BFS using a Queue - no parallelism
	//------------------------------------------------------------------------------------------------------------

	// Reset Nodes to unpassed
	for (int i = 0; i < MAXsize; i++) nodes[i].passed = false;

	// Find BFS using Queue
	cout << "BFS using Queue (normal)  : ";
	auto startBFSQ_NP = std::chrono::high_resolution_clock::now(); // Get time before BFS

	//Call BFS function
	if (FirstNodeLinksNum >= LastNodeLinksNum) BFS_Q(&nodes[0], false); // Top-down BFS
	else BFS_Q(&nodes[MAXsize - 1], false); // Bottom-up BFS

	auto endBFSQ_NP = std::chrono::high_resolution_clock::now(); // Get time after BFS
	cout << endl;

	// Calculate and display execution time
	auto executionTimeBFSQ_NP = std::chrono::duration_cast<std::chrono::microseconds>(endBFSQ_NP - startBFSQ_NP).count();

	//------------------------------------------------------------------------------------------------------------
	// BFS using a Queue - with parallelism
	//------------------------------------------------------------------------------------------------------------

	// Reset Nodes to unpassed
	for (int i = 0; i < MAXsize; i++) nodes[i].passed = false;

	// Find BFS using Queue
	cout << endl << "BFS using Queue (Parallel): ";
	auto startBFSQ = std::chrono::high_resolution_clock::now(); // Get time before BFS

	//Call BFS function
	if (FirstNodeLinksNum >= LastNodeLinksNum) BFS_Q(&nodes[0], true); // Top-down BFS
	else BFS_Q(&nodes[MAXsize - 1], true); // Bottom-up BFS

	auto endBFSQ = std::chrono::high_resolution_clock::now(); // Get time after BFS
	cout << endl;

	// Calculate and display execution time
	auto executionTimeBFSQ = std::chrono::duration_cast<std::chrono::microseconds>(endBFSQ - startBFSQ).count();

	//------------------------------------------------------------------------------------------------------------
	// BFS using Array - no parallelism
	//------------------------------------------------------------------------------------------------------------

	// Reset Nodes to unpassed
	for (int i = 0; i < MAXsize; i++) nodes[i].passed = false;

	// If first node has more links than last nodes, start with it
	// Otherwise, start with the last node
	cout << endl << "BFS using Array (normal)  : ";

	auto startBFSA_NP = std::chrono::high_resolution_clock::now(); // Get time before BFS

	if (FirstNodeLinksNum >= LastNodeLinksNum) BFS_A(&nodes[0], false); // Top-down BFS
	else BFS_A(&nodes[MAXsize - 1], false); // Bottom-up BFS

	auto endBFSA_NP = std::chrono::high_resolution_clock::now(); // Get time after BFS

	// Calculate and display execution time
	auto executionTimeBFSA_NP = std::chrono::duration_cast<std::chrono::microseconds>(endBFSA_NP - startBFSA_NP).count();

	//------------------------------------------------------------------------------------------------------------
 	// BFS using Array - with parallelism
	//------------------------------------------------------------------------------------------------------------

	// Reset Nodes to unpassed
	for (int i = 0; i < MAXsize; i++) nodes[i].passed = false;

	// If first node has more links than last nodes, start with it
	// Otherwise, start with the last node
	cout << endl << endl << "BFS using Array (Parallel): ";

	auto startBFSA = std::chrono::high_resolution_clock::now(); // Get time before BFS

	if (FirstNodeLinksNum >= LastNodeLinksNum) BFS_A(&nodes[0], true); // Top-down BFS
	else BFS_A(&nodes[MAXsize-1], true); // Bottom-up BFS

	auto endBFSA = std::chrono::high_resolution_clock::now(); // Get time after BFS
	cout << endl;

	// Calculate and display execution time
	auto executionTimeBFSA = std::chrono::duration_cast<std::chrono::microseconds>(endBFSA - startBFSA).count();

	//------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------

	cout << endl;
	cout << endl << "*Orders may differ due to parallization";
	cout << endl << "*Some values may be printed twice in parallel functions";
	cout << endl;

	// Print execution time for each case
	cout << endl << "------------------------------------------------------------------" << endl;
	cout << "Number of nodes in the network/graph (" << MAXsize << ") where each node may have up to (";
	cout << MAXlinks << ") links to other nodes" << endl;
	cout << "The number of edges is: (" << edges << ")" << endl << endl;

	cout << "*Page Rank Execution Time is: " << executionTimePR_NP << " micro-seconds" << endl; // (Sequential)
	//cout << "*Page Rank (Parallel) Execution Time is   : " << executionTimePR << " micro-seconds" << endl; // (Parallel)
	cout << endl;

	// Parallelized Page Rank has been removed because it is slower than its sequential counterpart

	//float PRn1 = executionTimePR_NP;
	//float PRn2 = executionTimePR;
	//float PRspeedup = PRn1 / PRn2;

	//cout << "*Page rank parallelism speedup is = " << PRspeedup << endl << endl;

	// BFS using queue execution time
	cout << "*BFS using Queue with no parallelism Execution Time is: " << executionTimeBFSQ_NP << " micro-seconds" << endl;
	cout << "*BFS using Queue with parallelism Execution Time is   : " << executionTimeBFSQ << " micro-seconds" << endl;

	// BFS using array execution time
	cout << "*BFS using Array with no parallelism Execution Time is: " << executionTimeBFSA_NP << " micro-seconds" << endl;
	cout << "*BFS using Array with parallelism Execution Time is   : " << executionTimeBFSA << " micro-seconds" << endl;

	// Assign execution times to variables
	float n1 = executionTimeBFSQ_NP;
	float n2 = executionTimeBFSA_NP;
	float n3 = executionTimeBFSQ;
	float n4 = executionTimeBFSA;

	// Calculate speedup
	float speedup1 = n1 / n3;
	float speedup2 = n2 / n4;
	float speedup3 = n1 / n2;
	float speedup4 = n3 / n4;

	cout << endl;
	printf("*Speedup (Queue normal/parallel)   = %2.6f -- (speedup of parallelizing Queue)\n", speedup1);
	printf("*Speedup (Array normal/parallel)   = %2.6f -- (speedup of parallelizing Array)\n", speedup2);
	printf("*Speedup (Queue/Array -- normal)   = %2.6f -- (speedup of using Array in place of Queue)\n", speedup3);
	printf("*Speedup (Queue/Array -- parallel) = %2.6f -- (speedup of using parallelized Array in place of parallelized Queue)\n\n", speedup4);

	cout << "-///------------------------------------------------------------------\\\\\\-" << endl;
	cout << "<<<---------------------------- Thank you! ---------------------------->>>" << endl;
	cout << "-\\\\\\------------------------------------------------------------------///-" << endl << endl;

	return 0;
}