#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Function to perform modular exponentiation (x^y % p)
long long power(long long x, unsigned long long y, long long p) {
    long long res = 1;
    x = x % p;
    while (y > 0) {
        if (y & 1)
            res = (__int128)res * x % p;
        y = y >> 1;
        x = (__int128)x * x % p;
    }
    return res;
}

// Miller-Rabin primality test for a single witness
bool millerTest(long long d, long long n) {
    long long a = 2 + rand() % (n - 4);
    long long x = power(a, d, n);

    if (x == 1 || x == n - 1)
        return true;

    while (d != n - 1) {
        x = (__int128)x * x % n;
        d *= 2;

        if (x == 1)
            return false;
        if (x == n - 1)
            return true;
    }
    return false;
}

// Main function to check if n is prime
bool isPrime(long long n, int k) {
    if (n <= 1 || n == 4)
        return false;
    if (n <= 3)
        return true;

    long long d = n - 1;
    while (d % 2 == 0)
        d /= 2;

    for (int i = 0; i < k; i++)
        if (!millerTest(d, n))
            return false;

    return true;
}

int main() {
    srand(time(0));
    long long n;

    std::cout << "--- Miller-Rabin Primality Test ---" << std::endl;
    std::cout << "This program tests if a given number is prime using the Miller-Rabin algorithm." << std::endl;
    std::cout << "The algorithm is probabilistic, but with a sufficient number of iterations (k=4)," << std::endl;
    std::cout << "the probability of a composite number being incorrectly identified as prime is extremely low." << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    while (true) {
        std::cout << "\nEnter a number to test (or enter 0 or a negative number to exit): ";
        std::cin >> n;

        if (n <= 0) {
            std::cout << "Exiting the program. Goodbye!" << std::endl;
            break;
        }

        if (isPrime(n, 4)) // k=4 is a good number of iterations for reasonable certainty
            std::cout << n << " is probably prime." << std::endl;
        else
            std::cout << n << " is composite." << std::endl;
    }

    return 0;
}
