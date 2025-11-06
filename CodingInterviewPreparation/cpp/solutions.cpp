// https://www.hackerrank.com/domains/cpp?filters%5Bsubdomains%5D%5B%5D=cpp-introduction&filters%5Bsubdomains%5D%5B%5D=cpp-strings&filters%5Bsubdomains%5D%5B%5D=classes&filters%5Bsubdomains%5D%5B%5D=stl&filters%5Bsubdomains%5D%5B%5D=inheritance&filters%5Bsubdomains%5D%5B%5D=cpp-debugging&filters%5Bsubdomains%5D%5B%5D=other-concepts

//============================================================================
// Say "Hello, World!" With C++
//============================================================================
/*
Problem: Print "Hello, World!" to stdout
Description: Basic output operation in C++
Input: None
Output: "Hello, World!"
*/

#include <iostream>
#include <cstdio>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}

//============================================================================
// Input and Output
//============================================================================
/*
Problem: Read three integers and output their sum
Description: Practice input/output operations
Input: Three integers a, b, c
Output: Sum of a + b + c
*/

#include <iostream>
using namespace std;

int main() {
    int a, b, c;
    cin >> a >> b >> c;
    cout << a + b + c << endl;
    return 0;
}

//============================================================================
// Basic Data Types
//============================================================================
/*
Problem: Read and output different data types
Description: Work with int, long, char, float, and double
Input: int, long, char, float, double on separate lines
Output: Each value on separate lines with proper formatting
*/

#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    int i;
    long l;
    char c;
    float f;
    double d;
    
    cin >> i >> l >> c >> f >> d;
    
    cout << i << endl;
    cout << l << endl;
    cout << c << endl;
    cout << fixed << setprecision(3) << f << endl;
    cout << fixed << setprecision(9) << d << endl;
    
    return 0;
}

//============================================================================
// Conditional Statements
//============================================================================
/*
Problem: Print number word for 1-9, or "Greater than 9"
Description: Use if-else statements
Input: Integer n
Output: Word representation or "Greater than 9"
*/

#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    if (n == 1) cout << "one";
    else if (n == 2) cout << "two";
    else if (n == 3) cout << "three";
    else if (n == 4) cout << "four";
    else if (n == 5) cout << "five";
    else if (n == 6) cout << "six";
    else if (n == 7) cout << "seven";
    else if (n == 8) cout << "eight";
    else if (n == 9) cout << "nine";
    else cout << "Greater than 9";
    
    return 0;
}

//============================================================================
// For Loop
//============================================================================
/*
Problem: Print number words and even/odd for range
Description: For numbers 1-9 print word, for n>9 print even/odd
Input: Two integers a and b
Output: For each number from a to b, print word or even/odd
*/

#include <iostream>
using namespace std;

int main() {
    int a, b;
    cin >> a >> b;
    
    string numbers[] = {"", "one", "two", "three", "four", "five", 
                        "six", "seven", "eight", "nine"};
    
    for (int i = a; i <= b; i++) {
        if (i <= 9) {
            cout << numbers[i] << endl;
        } else {
            cout << (i % 2 == 0 ? "even" : "odd") << endl;
        }
    }
    
    return 0;
}

//============================================================================
// Functions
//============================================================================
/*
Problem: Find maximum of four integers
Description: Write a function max_of_four
Input: Four integers
Output: Maximum value
*/

#include <iostream>
using namespace std;

int max_of_four(int a, int b, int c, int d) {
    int max_val = a;
    if (b > max_val) max_val = b;
    if (c > max_val) max_val = c;
    if (d > max_val) max_val = d;
    return max_val;
}

int main() {
    int a, b, c, d;
    cin >> a >> b >> c >> d;
    cout << max_of_four(a, b, c, d) << endl;
    return 0;
}

//============================================================================
// Pointer
//============================================================================
/*
Problem: Modify values using pointers
Description: Update *a to sum and *b to absolute difference
Input: Two integers
Output: Modified values
*/

#include <iostream>
using namespace std;

void update(int *a, int *b) {
    int temp_a = *a;
    *a = temp_a + *b;
    *b = abs(temp_a - *b);
}

int main() {
    int a, b;
    cin >> a >> b;
    update(&a, &b);
    cout << a << endl << b << endl;
    return 0;
}

//============================================================================
// Arrays Introduction
//============================================================================
/*
Problem: Reverse an array
Description: Read N integers and print in reverse order
Input: N, then N integers
Output: Array in reverse order
*/

#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    int arr[n];
    
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    
    for (int i = n - 1; i >= 0; i--) {
        cout << arr[i] << " ";
    }
    
    return 0;
}

//============================================================================
// Variable Sized Arrays
//============================================================================
/*
Problem: Work with variable-sized arrays
Description: Create n arrays of variable sizes and answer queries
Input: n (arrays), q (queries), n arrays, q queries
Output: Answer for each query
*/

#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, q;
    cin >> n >> q;
    
    vector<vector<int>> arrays(n);
    
    // Read arrays
    for (int i = 0; i < n; i++) {
        int k;
        cin >> k;
        arrays[i].resize(k);
        for (int j = 0; j < k; j++) {
            cin >> arrays[i][j];
        }
    }
    
    // Answer queries
    for (int i = 0; i < q; i++) {
        int a, b;
        cin >> a >> b;
        cout << arrays[a][b] << endl;
    }
    
    return 0;
}

//============================================================================
// StringStream
//============================================================================
/*
Problem: Parse comma-separated integers from string
Description: Use stringstream to parse string
Input: String with comma-separated integers
Output: Vector of integers
*/

#include <sstream>
#include <vector>
#include <iostream>
using namespace std;

vector<int> parseInts(string str) {
    vector<int> result;
    stringstream ss(str);
    int num;
    char ch;
    
    while (ss >> num) {
        result.push_back(num);
        ss >> ch; // Read comma
    }
    
    return result;
}

int main() {
    string str;
    cin >> str;
    vector<int> integers = parseInts(str);
    for(int i = 0; i < integers.size(); i++) {
        cout << integers[i] << "\n";
    }
    return 0;
}

//============================================================================
// Strings
//============================================================================
/*
Problem: Concatenate and manipulate strings
Description: Output sizes, concatenation, and swap first characters
Input: Two strings a and b
Output: Sizes, concatenated string, strings with first chars swapped
*/

#include <iostream>
#include <string>
using namespace std;

int main() {
    string a, b;
    cin >> a >> b;
    
    // Print sizes
    cout << a.size() << " " << b.size() << endl;
    
    // Print concatenation
    cout << a + b << endl;
    
    // Swap first characters
    char temp = a[0];
    a[0] = b[0];
    b[0] = temp;
    
    cout << a << " " << b << endl;
    
    return 0;
}

//============================================================================
// Structs
//============================================================================
/*
Problem: Create and use a struct
Description: Define Student struct and process data
Input: Student data (first_name, last_name, age, standard)
Output: Student information
*/

#include <iostream>
using namespace std;

struct Student {
    int age;
    string first_name;
    string last_name;
    int standard;
};

int main() {
    Student st;
    
    cin >> st.age >> st.first_name >> st.last_name >> st.standard;
    
    cout << st.age << " " << st.first_name << " " << st.last_name << " " << st.standard;
    
    return 0;
}

//============================================================================
// Class
//============================================================================
/*
Problem: Create a basic class with getter/setter
Description: Create Student class with private members
Input: Student data
Output: Formatted student information
*/

#include <iostream>
#include <sstream>
using namespace std;

class Student {
private:
    int age;
    int standard;
    string first_name;
    string last_name;
    
public:
    void set_age(int a) { age = a; }
    void set_standard(int s) { standard = s; }
    void set_first_name(string fn) { first_name = fn; }
    void set_last_name(string ln) { last_name = ln; }
    
    int get_age() { return age; }
    int get_standard() { return standard; }
    string get_first_name() { return first_name; }
    string get_last_name() { return last_name; }
    
    string to_string() {
        stringstream ss;
        ss << age << "," << first_name << "," << last_name << "," << standard;
        return ss.str();
    }
};

int main() {
    int age, standard;
    string first_name, last_name;
    
    cin >> age >> first_name >> last_name >> standard;
    
    Student st;
    st.set_age(age);
    st.set_standard(standard);
    st.set_first_name(first_name);
    st.set_last_name(last_name);
    
    cout << st.get_age() << "\n";
    cout << st.get_last_name() << ", " << st.get_first_name() << "\n";
    cout << st.get_standard() << "\n";
    cout << "\n";
    cout << st.to_string();
    
    return 0;
}

//============================================================================
// Classes and Objects
//============================================================================
/*
Problem: Track student scores and calculate totals
Description: Create Student class with score tracking
Input: 5 test scores for each of 5 students, then query student
Output: Total score of queried student
*/

#include <iostream>
#include <vector>
using namespace std;

class Student {
private:
    vector<int> scores;
    
public:
    void input() {
        for (int i = 0; i < 5; i++) {
            int score;
            cin >> score;
            scores.push_back(score);
        }
    }
    
    int calculateTotalScore() {
        int total = 0;
        for (int score : scores) {
            total += score;
        }
        return total;
    }
};

int main() {
    int n;
    cin >> n;
    
    Student students[n];
    
    for (int i = 0; i < n; i++) {
        students[i].input();
    }
    
    int query_student;
    cin >> query_student;
    
    cout << students[query_student].calculateTotalScore();
    
    return 0;
}

//============================================================================
// Box It!
//============================================================================
/*
Problem: Create Box class with operators
Description: Implement constructors, operators, and methods for Box class
Input: Various box operations
Output: Box calculations
*/

#include <iostream>
using namespace std;

class Box {
private:
    long long l, b, h;
    
public:
    Box() : l(0), b(0), h(0) {}
    Box(int length, int breadth, int height) : l(length), b(breadth), h(height) {}
    Box(const Box& B) : l(B.l), b(B.b), h(B.h) {}
    
    int getLength() { return l; }
    int getBreadth() { return b; }
    int getHeight() { return h; }
    
    long long CalculateVolume() {
        return l * b * h;
    }
    
    bool operator<(Box& B) {
        if (l < B.l) return true;
        if (b < B.b && l == B.l) return true;
        if (h < B.h && b == B.b && l == B.l) return true;
        return false;
    }
    
    friend ostream& operator<<(ostream& out, Box& B);
};

ostream& operator<<(ostream& out, Box& B) {
    out << B.l << " " << B.b << " " << B.h;
    return out;
}

//============================================================================
// Vector-Sort
//============================================================================
/*
Problem: Sort a vector of integers
Description: Read integers, sort them, and output
Input: N, then N integers
Output: Sorted integers
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> v(n);
    for (int i = 0; i < n; i++) {
        cin >> v[i];
    }
    
    sort(v.begin(), v.end());
    
    for (int i = 0; i < n; i++) {
        cout << v[i] << " ";
    }
    
    return 0;
}

//============================================================================
// Vector-Erase
//============================================================================
/*
Problem: Erase elements from vector
Description: Erase one element and a range
Input: N integers, position to erase, range to erase
Output: Size and remaining elements
*/

#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> v(n);
    for (int i = 0; i < n; i++) {
        cin >> v[i];
    }
    
    int pos;
    cin >> pos;
    v.erase(v.begin() + pos - 1);
    
    int a, b;
    cin >> a >> b;
    v.erase(v.begin() + a - 1, v.begin() + b - 1);
    
    cout << v.size() << endl;
    for (int x : v) {
        cout << x << " ";
    }
    
    return 0;
}

//============================================================================
// Lower Bound-STL
//============================================================================
/*
Problem: Find lower_bound in sorted vector
Description: Use lower_bound to find elements
Input: N integers, Q queries
Output: "Yes position" or "No position" for each query
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> v(n);
    for (int i = 0; i < n; i++) {
        cin >> v[i];
    }
    
    int q;
    cin >> q;
    
    while (q--) {
        int x;
        cin >> x;
        
        auto it = lower_bound(v.begin(), v.end(), x);
        
        if (it != v.end() && *it == x) {
            cout << "Yes " << (it - v.begin() + 1) << endl;
        } else {
            cout << "No " << (it - v.begin() + 1) << endl;
        }
    }
    
    return 0;
}

//============================================================================
// Sets-STL
//============================================================================
/*
Problem: Perform set operations
Description: Add, erase, and find elements in a set
Input: Q queries with operations
Output: "Yes" or "No" for find queries
*/

#include <iostream>
#include <set>
using namespace std;

int main() {
    int q;
    cin >> q;
    
    set<int> s;
    
    while (q--) {
        int type, x;
        cin >> type >> x;
        
        if (type == 1) {
            s.insert(x);
        } else if (type == 2) {
            s.erase(x);
        } else {
            if (s.find(x) != s.end()) {
                cout << "Yes" << endl;
            } else {
                cout << "No" << endl;
            }
        }
    }
    
    return 0;
}

//============================================================================
// Maps-STL
//============================================================================
/*
Problem: Maintain student marks using map
Description: Add, erase, and query marks
Input: Q queries with operations
Output: Marks for query type 3
*/

#include <iostream>
#include <map>
using namespace std;

int main() {
    int q;
    cin >> q;
    
    map<string, int> m;
    
    while (q--) {
        int type;
        string name;
        cin >> type >> name;
        
        if (type == 1) {
            int marks;
            cin >> marks;
            m[name] += marks;
        } else if (type == 2) {
            m.erase(name);
        } else {
            cout << m[name] << endl;
        }
    }
    
    return 0;
}

//============================================================================
// Print Pretty
//============================================================================
/*
Problem: Format output with different bases and precision
Description: Print in hex, decimal with indicators, and fixed precision
Input: Three doubles A, B, C
Output: Formatted output
*/

#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    int A;
    double B, C;
    cin >> A >> B >> C;
    
    // Line 1: Hex with 0x prefix
    cout << hex << showbase << nouppercase << A << endl;
    
    // Line 2: Decimal with right align, sign, precision 2, width 15
    cout << dec << fixed << setprecision(2) << showpos << setfill('_') << setw(15) << B << endl;
    
    // Line 3: Scientific notation, precision 9, uppercase
    cout << scientific << noshowpos << uppercase << setprecision(9) << C << endl;
    
    return 0;
}

//============================================================================
// Deque-STL
//============================================================================
/*
Problem: Find maximum in each subarray of size K
Description: Use deque to efficiently track maximums
Input: N (array size), K (subarray size), array elements
Output: Maximum of each K-sized subarray
*/

#include <iostream>
#include <deque>
using namespace std;

void printKMax(int arr[], int n, int k) {
    deque<int> dq;
    
    // Process first k elements
    for (int i = 0; i < k; i++) {
        while (!dq.empty() && arr[i] >= arr[dq.back()]) {
            dq.pop_back();
        }
        dq.push_back(i);
    }
    
    // Process remaining elements
    for (int i = k; i < n; i++) {
        cout << arr[dq.front()] << " ";
        
        // Remove elements outside window
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        // Remove smaller elements
        while (!dq.empty() && arr[i] >= arr[dq.back()]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
    }
    
    cout << arr[dq.front()] << endl;
}

int main() {
    int t;
    cin >> t;
    
    while (t--) {
        int n, k;
        cin >> n >> k;
        
        int arr[n];
        for (int i = 0; i < n; i++) {
            cin >> arr[i];
        }
        
        printKMax(arr, n, k);
    }
    
    return 0;
}

//============================================================================
// Inheritance Introduction
//============================================================================
/*
Problem: Demonstrate basic inheritance
Description: Triangle inherits from Triangle
Input: None (predefined)
Output: Triangle output through inheritance
*/

#include <iostream>
using namespace std;

class Triangle {
public:
    void triangle() {
        cout << "I am a triangle\n";
    }
};

class Isosceles : public Triangle {
public:
    void isosceles() {
        cout << "I am an isosceles triangle\n";
    }
};

int main() {
    Isosceles isc;
    isc.isosceles();
    isc.triangle();
    return 0;
}

//============================================================================
// Rectangle Area
//============================================================================
/*
Problem: Calculate rectangle area with classes
Description: Use inheritance to create Rectangle from Display
Input: Width and height
Output: Area
*/

#include <iostream>
using namespace std;

class Rectangle {
protected:
    int width;
    int height;
    
public:
    void read_input() {
        cin >> width >> height;
    }
    
    void display() {
        cout << width * height << endl;
    }
};

class RectangleArea : public Rectangle {
public:
    void display() {
        cout << width * height << endl;
    }
};

int main() {
    RectangleArea r_area;
    r_area.read_input();
    r_area.display();
    return 0;
}

//============================================================================
// Multi Level Inheritance
//============================================================================
/*
Problem: Demonstrate multi-level inheritance
Description: Isosceles -> Equilateral inheritance chain
Input: None
Output: Triangle type information
*/

#include <iostream>
using namespace std;

class Triangle {
public:
    void triangle() {
        cout << "I am a triangle\n";
    }
};

class Isosceles : public Triangle {
public:
    void isosceles() {
        cout << "I am an isosceles triangle\n";
    }
};

class Equilateral : public Isosceles {
public:
    void equilateral() {
        cout << "I am an equilateral triangle\n";
    }
};

int main() {
    Equilateral eqr;
    eqr.equilateral();
    eqr.isosceles();
    eqr.triangle();
    return 0;
}

//============================================================================
// Inherited Code (Exception Handling)
//============================================================================
/*
Problem: Handle exceptions in inherited code
Description: Create BadLengthException and catch it
Input: String lengths
Output: String or error message
*/

#include <iostream>
#include <string>
#include <exception>
using namespace std;

class BadLengthException : public exception {
private:
    int n;
    
public:
    BadLengthException(int length) : n(length) {}
    
    int what() {
        return n;
    }
};

bool checkUsername(string username) {
    bool isValid = true;
    int n = username.length();
    if (n < 5) {
        throw BadLengthException(n);
    }
    for (int i = 0; i < n - 1; i++) {
        if (username[i] == 'w' && username[i + 1] == 'w') {
            isValid = false;
        }
    }
    return isValid;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        string username;
        cin >> username;
        try {
            bool isValid = checkUsername(username);
            if (isValid) {
                cout << "Valid" << '\n';
            } else {
                cout << "Invalid" << '\n';
            }
        } catch (BadLengthException e) {
            cout << "Too short: " << e.what() << '\n';
        }
    }
    return 0;
}

//============================================================================
// Exceptional Server
//============================================================================
/*
Problem: Handle multiple exception types
Description: Catch different exception types and handle appropriately
Input: Server operations
Output: Exception messages or results
*/

#include <iostream>
#include <exception>
#include <string>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace std;

class Server {
private:
    static int load;
    
public:
    static int compute(long long A, long long B) {
        load += 1;
        if (A < 0) {
            throw invalid_argument("A is negative");
        }
        vector<int> v(A, 0);
        int real = -1, cmplx = sqrt(-1);
        if (B == 0) throw 0;
        real = (A / B) * real;
        int ans = v.at(B);
        return real + A - B * ans;
    }
};

int Server::load = 0;

int main() {
    int T;
    cin >> T;
    while (T--) {
        long long A, B;
        cin >> A >> B;
        try {
            cout << Server::compute(A, B) << endl;
        } catch (bad_alloc& error) {
            cout << "Not enough memory" << endl;
        } catch (exception& error) {
            cout << "Exception: " << error.what() << endl;
        } catch (...) {
            cout << "Other Exception" << endl;
        }
    }
    return 0;
}

//============================================================================
// Virtual Functions
//============================================================================
/*
Problem: Implement virtual functions for polymorphism
Description: Create Person base class with Professor and Student derived
Input: Person data
Output: Person information based on type
*/

#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Person {
protected:
    string name;
    int age;
    
public:
    virtual void getdata() = 0;
    virtual void putdata() = 0;
};

class Professor : public Person {
private:
    int publications;
    static int id_counter;
    int cur_id;
    
public:
    Professor() {
        cur_id = ++id_counter;
    }
    
    void getdata() {
        cin >> name >> age >> publications;
    }
    
    void putdata() {
        cout << name << " " << age << " " << publications << " " << cur_id << endl;
    }
};

class Student : public Person {
private:
    int marks[6];
    static int id_counter;
    int cur_id;
    
public:
    Student() {
        cur_id = ++id_counter;
    }
    
    void getdata() {
        cin >> name >> age;
        for (int i = 0; i < 6; i++) {
            cin >> marks[i];
        }
    }
    
    void putdata() {
        int total = 0;
        for (int i = 0; i < 6; i++) {
            total += marks[i];
        }
        cout << name << " " << age << " " << total << " " << cur_id << endl;
    }
};

int Professor::id_counter = 0;
int Student::id_counter = 0;

int main() {
    int n, val;
    cin >> n;
    Person *per[n];
    
    for (int i = 0; i < n; i++) {
        cin >> val;
        if (val == 1) {
            per[i] = new Professor;
        } else {
            per[i] = new Student;
        }
        per[i]->getdata();
    }
    
    for (int i = 0; i < n; i++) {
        per[i]->putdata();
    }
    
    return 0;
}

//============================================================================
// Abstract Classes - Polymorphism
//============================================================================
/*
Problem: Implement abstract class for caching
Description: Create LRUCache with abstract Cache base
Input: Cache operations
Output: Cache results
*/

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <set>
#include <cassert>
using namespace std;

struct Node {
   Node* next;
   Node* prev;
   int value;
   int key;
   Node(Node* p, Node* n, int k, int val) : prev(p), next(n), key(k), value(val) {}
   Node(int k, int val) : prev(NULL), next(NULL), key(k), value(val) {}
};

class Cache {
protected: 
   map<int, Node*> mp;
   int cp;
   Node* tail;
   Node* head;
   
   virtual void set(int, int) = 0;
   virtual int get(int) = 0;
};

class LRUCache : public Cache {
public:
    LRUCache(int capacity) {
        cp = capacity;
        head = NULL;
        tail = NULL;
    }
    
    void set(int key, int value) {
        if (mp.find(key) != mp.end()) {
            Node* node = mp[key];
            node->value = value;
            
            if (node != head) {
                if (node == tail) {
                    tail = tail->prev;
                    tail->next = NULL;
                } else {
                    node->prev->next = node->next;
                    node->next->prev = node->prev;
                }
                
                node->next = head;
                node->prev = NULL;
                head->prev = node;
                head = node;
            }
        } else {
            Node* node = new Node(key, value);
            
            if (mp.size() >= cp) {
                mp.erase(tail->key);
                
                if (head == tail) {
                    head = node;
                    tail = node;
                } else {
                    Node* temp = tail;
                    tail = tail->prev;
                    tail->next = NULL;
                    delete temp;
                    
                    node->next = head;
                    head->prev = node;
                    head = node;
                }
            } else {
                if (head == NULL) {
                    head = node;
                    tail = node;
                } else {
                    node->next = head;
                    head->prev = node;
                    head = node;
                }
            }
            
            mp[key] = node;
        }
    }
    
    int get(int key) {
        if (mp.find(key) != mp.end()) {
            Node* node = mp[key];
            
            if (node != head) {
                if (node == tail) {
                    tail = tail->prev;
                    tail->next = NULL;
                } else {
                    node->prev->next = node->next;
                    node->next->prev = node->prev;
                }
                
                node->next = head;
                node->prev = NULL;
                head->prev = node;
                head = node;
            }
            
            return node->value;
        }
        return -1;
    }
};

//============================================================================
// C++ Class Templates
//============================================================================
/*
Problem: Create a generic class template
Description: Implement AddElements template class
Input: Different data types
Output: Sum of elements
*/

#include <iostream>
using namespace std;

template <class T>
class AddElements {
    T element;
    
public:
    AddElements(T arg) {
        element = arg;
    }
    
    T add(T arg) {
        return element + arg;
    }
};

// Template specialization for concatenating strings
template <>
class AddElements<string> {
    string element;
    
public:
    AddElements(string arg) {
        element = arg;
    }
    
    string concatenate(string arg) {
        return element + arg;
    }
};

//============================================================================
// Preprocessor Solution
//============================================================================
/*
Problem: Use preprocessor directives
Description: Define macros for operations
Input: Numbers
Output: Results using macros
*/

#include <iostream>
using namespace std;

#define io(v) cin >> v
#define INF 100000
#define FUNCTION(name, operator) inline void name(int& a, int b) { if (b operator a) a = b; }
#define toStr(x) #x
#define foreach(v, i) for (int i = 0; i < v.size(); i++)
#define minimum(a, b) ((a) < (b) ? (a) : (b))
#define maximum(a, b) ((a) > (b) ? (a) : (b))

FUNCTION(minimum, <)
FUNCTION(maximum, >)

//============================================================================
// Operator Overloading
//============================================================================
/*
Problem: Overload operators for complex numbers
Description: Implement +, - operators for Complex class
Input: Complex number operations
Output: Results
*/

#include <iostream>
using namespace std;

class Complex {
public:
    int a, b;
    
    void input(string s) {
        int v1 = 0;
        int i = 0;
        while (s[i] != '+') {
            v1 = v1 * 10 + s[i] - '0';
            i++;
        }
        while (s[i] == ' ' || s[i] == '+' || s[i] == 'i') {
            i++;
        }
        int v2 = 0;
        while (i < s.length()) {
            v2 = v2 * 10 + s[i] - '0';
            i++;
        }
        a = v1;
        b = v2;
    }
};

Complex operator+(const Complex& x, const Complex& y) {
    Complex result;
    result.a = x.a + y.a;
    result.b = x.b + y.b;
    return result;
}

ostream& operator<<(ostream& os, const Complex& c) {
    return os << c.a << "+i" << c.b;
}

//============================================================================
// Overload Operators
//============================================================================
/*
Problem: Overload operators for Matrix class
Description: Implement + operator for matrices
Input: Matrix data
Output: Sum of matrices
*/

#include <iostream>
#include <vector>
using namespace std;

class Matrix {
public:
    vector<vector<int>> a;
    
    Matrix operator+(const Matrix& other) {
        Matrix result;
        int n = a.size();
        int m = a[0].size();
        result.a.resize(n);
        
        for (int i = 0; i < n; i++) {
            result.a[i].resize(m);
            for (int j = 0; j < m; j++) {
                result.a[i][j] = a[i][j] + other.a[i][j];
            }
        }
        
        return result;
    }
};

//============================================================================
// Attending Workshops
//============================================================================
/*
Problem: Select maximum non-overlapping workshops
Description: Activity selection problem
Input: N workshops with start and end times
Output: Maximum workshops that can be attended
*/

#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

struct Workshop {
    int start;
    int duration;
    int end;
};

struct Available_Workshops {
    int n;
    Workshop* workshops;
};

Available_Workshops* initialize(int start_time[], int duration[], int n) {
    Available_Workshops* aws = new Available_Workshops;
    aws->n = n;
    aws->workshops = new Workshop[n];
    
    for (int i = 0; i < n; i++) {
        aws->workshops[i].start = start_time[i];
        aws->workshops[i].duration = duration[i];
        aws->workshops[i].end = start_time[i] + duration[i];
    }
    
    return aws;
}

int CalculateMaxWorkshops(Available_Workshops* ptr) {
    sort(ptr->workshops, ptr->workshops + ptr->n, 
         [](const Workshop& a, const Workshop& b) {
             return a.end < b.end;
         });
    
    int count = 1;
    int last_end = ptr->workshops[0].end;
    
    for (int i = 1; i < ptr->n; i++) {
        if (ptr->workshops[i].start >= last_end) {
            count++;
            last_end = ptr->workshops[i].end;
        }
    }
    
    return count;
}

//============================================================================
// C++ Class Template Specialization
//============================================================================
/*
Problem: Create specialized template for enum
Description: Implement Traits template with enum specialization
Input: None (testing framework)
Output: Enum name string
*/

#include <iostream>
using namespace std;

enum class Fruit { apple, orange, pear };
enum class Color { red, green, orange };

template <typename T>
struct Traits;

template <>
struct Traits<Color> {
    static string name(int index) {
        switch(index) {
            case 0: return "red";
            case 1: return "green";
            case 2: return "orange";
        }
        return "unknown";
    }
};

template <>
struct Traits<Fruit> {
    static string name(int index) {
        switch(index) {
            case 0: return "apple";
            case 1: return "orange";
            case 2: return "pear";
        }
        return "unknown";
    }
};

//============================================================================
// C++ Variadics
//============================================================================
/*
Problem: Use variadic templates
Description: Implement reversed function with variadic templates
Input: Variable number of arguments
Output: Arguments in reverse order
*/

#include <iostream>
using namespace std;

template <bool...> struct bool_pack;

template <bool... v>
using all_true = is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

template <int... Args>
struct IntList;

template <int N, typename List>
struct Prepend;

template <int N, int... Args>
struct Prepend<N, IntList<Args...>> {
    using type = IntList<N, Args...>;
};

template <int N>
struct Range {
    using type = typename Prepend<N, typename Range<N - 1>::type>::type;
};

template <>
struct Range<0> {
    using type = IntList<>;
};

template <typename F, int... Args>
void execute(F f, IntList<Args...>) {
    int dummy[] = { (f(Args), 0)... };
}

template <typename F, int N>
void reversed_binary_value(F f) {
    execute(f, typename Range<N>::type());
}

//============================================================================
// Bit Array
//============================================================================
/*
Problem: Find maximum XOR in subarray
Description: Efficient algorithm using bit manipulation
Input: N, S, P, Q then Q queries
Output: Maximum XOR for each query
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
using namespace std;

int main() {
    int n, s, p, q;
    cin >> n >> s >> p >> q;
    
    vector<long long> arr(n);
    arr[0] = s % p;
    
    for (int i = 1; i < n; i++) {
        arr[i] = (arr[i - 1] * s + p) % q;
    }
    
    long long max_xor = 0;
    set<long long> unique_elements;
    
    for (int i = 0; i < n; i++) {
        unique_elements.insert(arr[i]);
        
        for (long long val : unique_elements) {
            max_xor = max(max_xor, arr[i] ^ val);
        }
    }
    
    cout << max_xor << endl;
    
    return 0;
}

//============================================================================
// Hotel Prices
//============================================================================
/*
Problem: Calculate hotel prices with inheritance
Description: Different room types with different pricing
Input: Room data
Output: Total cost
*/

#include <iostream>
#include <vector>
using namespace std;

class HotelRoom {
public:
    HotelRoom(int bedrooms, int bathrooms) 
        : bedrooms_(bedrooms), bathrooms_(bathrooms) {}
    
    virtual int get_price() {
        return 50 * bedrooms_ + 100 * bathrooms_;
    }
    
private:
    int bedrooms_;
    int bathrooms_;
};

class HotelApartment : public HotelRoom {
public:
    HotelApartment(int bedrooms, int bathrooms) 
        : HotelRoom(bedrooms, bathrooms) {}

    int get_price() {
        return HotelRoom::get_price() + 100;
    }
};

//============================================================================
// Cpp exception handling
//============================================================================
/*
Problem: Handle exceptions properly
Description: Catch and rethrow exceptions
Input: Server code
Output: Exception messages
*/

#include <iostream>
#include <stdexcept>
using namespace std;

int largest_proper_divisor(int n) {
    if (n == 0) {
        throw invalid_argument("largest proper divisor is not defined for n=0");
    }
    if (n == 1) {
        throw invalid_argument("largest proper divisor is not defined for n=1");
    }
    for (int i = n / 2; i >= 1; --i) {
        if (n % i == 0) {
            return i;
        }
    }
    return -1;
}

void process_input(int n) {
    try {
        int d = largest_proper_divisor(n);
        cout << "result=" << d << endl;
    } catch (const invalid_argument& e) {
        cout << e.what() << endl;
    }
    cout << "returning control flow to caller" << endl;
}

//============================================================================
// Messages Order
//============================================================================
/*
Problem: Sort messages by order received
Description: Maintain message order using comparison
Input: Messages from different senders
Output: Sorted messages
*/

#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

class Message {
private:
    string text;
    int order;
    static int counter;
    
public:
    Message() {
        order = counter++;
    }
    
    Message(const string& text) : text(text) {
        order = counter++;
    }
    
    const string& get_text() const {
        return text;
    }
    
    bool operator<(const Message& other) const {
        return order < other.order;
    }
};

int Message::counter = 0;

class MessageFactory {
public:
    MessageFactory() {}
    
    Message create_message(const string& text) {
        return Message(text);
    }
};

//============================================================================
// Accessing Inherited Functions
//============================================================================
/*
Problem: Access functions from multiple inheritance levels
Description: Call specific versions of overridden functions
Input: Sequence of operations
Output: Result using specific function versions
*/

#include <iostream>
using namespace std;

class A {
public:
    A() {
        callA = 0;
    }
    
private:
    int callA;
    void inc() {
        callA++;
    }

protected:
    void func(int & a) {
        a = a * 2;
        inc();
    }
    
public:
    int getA() {
        return callA;
    }
};

class B {
public:
    B() {
        callB = 0;
    }
    
private:
    int callB;
    void inc() {
        callB++;
    }
    
protected:
    void func(int & a) {
        a = a * 3;
        inc();
    }
    
public:
    int getB() {
        return callB;
    }
};

class C {
public:
    C() {
        callC = 0;
    }
    
private:
    int callC;
    void inc() {
        callC++;
    }
    
protected:
    void func(int & a) {
        a = a * 5;
        inc();
    }
    
public:
    int getC() {
        return callC;
    }
};

class D : public A, public B, public C {
    int val;
    
public:
    D() {
        val = 1;
    }

    void update_val(int new_val) {
        int operations = new_val;
        
        while (operations > 0) {
            if (operations % 5 == 0) {
                C::func(val);
                operations /= 5;
            } else if (operations % 3 == 0) {
                B::func(val);
                operations /= 3;
            } else if (operations % 2 == 0) {
                A::func(val);
                operations /= 2;
            } else {
                break;
            }
        }
    }

    void check(int new_val) {
        update_val(new_val);
        cout << "Value = " << val << endl
             << "A's func called " << getA() << " times" << endl
             << "B's func called " << getB() << " times" << endl
             << "C's func called " << getC() << " times" << endl;
    }
};

//============================================================================
// Magic Spells (Advanced - Dynamic Programming + LCS)
//============================================================================
/*
Problem: Find longest common subsequence for spell matching
Description: Use LCS algorithm for pattern matching
Input: Spells
Output: Spell results
*/

#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Spell {
private:
    string scrollName;
    
public:
    Spell(): scrollName("") { }
    Spell(string name): scrollName(name) { }
    virtual ~Spell() { }
    string revealScrollName() {
        return scrollName;
    }
};

class Fireball : public Spell {
private:
    int power;
    
public:
    Fireball(int power): power(power) { }
    void revealFirepower() {
        cout << "Fireball: " << power << endl;
    }
};

class Frostbite : public Spell {
private:
    int power;
    
public:
    Frostbite(int power): power(power) { }
    void revealFrostpower() {
        cout << "Frostbite: " << power << endl;
    }
};

class Thunderstorm : public Spell {
private:
    int power;
    
public:
    Thunderstorm(int power): power(power) { }
    void revealThunderpower() {
        cout << "Thunderstorm: " << power << endl;
    }
};

class Waterbolt : public Spell {
private:
    int power;
    
public:
    Waterbolt(int power): power(power) { }
    void revealWaterpower() {
        cout << "Waterbolt: " << power << endl;
    }
};

class SpellJournal {
public:
    static string journal;
    static string read() {
        return journal;
    }
};

string SpellJournal::journal = "";

void counterspell(Spell *spell) {
    if (Fireball* fb = dynamic_cast<Fireball*>(spell)) {
        fb->revealFirepower();
    } else if (Frostbite* frost = dynamic_cast<Frostbite*>(spell)) {
        frost->revealFrostpower();
    } else if (Thunderstorm* thunder = dynamic_cast<Thunderstorm*>(spell)) {
        thunder->revealThunderpower();
    } else if (Waterbolt* water = dynamic_cast<Waterbolt*>(spell)) {
        water->revealWaterpower();
    } else {
        string spellName = spell->revealScrollName();
        string journal = SpellJournal::read();
        
        int n = spellName.length();
        int m = journal.length();
        
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (spellName[i - 1] == journal[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        cout << dp[n][m] << endl;
    }
}

//============================================================================
// Attribute Parser (Complex String Parsing)
//============================================================================
/*
Problem: Parse HRML tags and answer queries
Description: Build tag hierarchy and query attributes
Input: N lines of HRML, Q queries
Output: Attribute values or "Not Found!"
*/

#include <iostream>
#include <map>
#include <vector>
#include <sstream>
using namespace std;

int main() {
    int n, q;
    cin >> n >> q;
    cin.ignore();
    
    map<string, string> tagMap;
    vector<string> tagStack;
    
    // Parse HRML
    for (int i = 0; i < n; i++) {
        string line;
        getline(cin, line);
        
        if (line.find("</") != string::npos) {
            // Closing tag
            if (!tagStack.empty()) {
                tagStack.pop_back();
            }
        } else {
            // Opening tag
            stringstream ss(line);
            string token;
            vector<string> tokens;
            
            while (ss >> token) {
                tokens.push_back(token);
            }
            
            // Extract tag name
            string tagName = tokens[0].substr(1);
            if (tagName.back() == '>') {
                tagName.pop_back();
            }
            
            tagStack.push_back(tagName);
            
            // Build full path
            string fullPath;
            for (int j = 0; j < tagStack.size(); j++) {
                if (j > 0) fullPath += ".";
                fullPath += tagStack[j];
            }
            
            // Parse attributes
            for (size_t j = 1; j < tokens.size(); j++) {
                if (tokens[j].find('=') != string::npos) {
                    size_t eqPos = tokens[j].find('=');
                    string attrName = tokens[j].substr(0, eqPos);
                    
                    string attrValue = tokens[j].substr(eqPos + 1);
                    if (j + 1 < tokens.size() && tokens[j + 1].back() == '>') {
                        attrValue += " " + tokens[j + 1];
                    }
                    
                    // Remove quotes and trailing >
                    attrValue.erase(remove(attrValue.begin(), attrValue.end(), '"'), attrValue.end());
                    if (attrValue.back() == '>') {
                        attrValue.pop_back();
                    }
                    string key = fullPath + "~" + attrName;
                    tagMap[key] = attrValue;
                }
            }
        }
    }
    
    // Answer queries
    for (int i = 0; i < q; i++) {
        string query;
        getline(cin, query);
        
        if (tagMap.find(query) != tagMap.end()) {
            cout << tagMap[query] << endl;
        } else {
            cout << "Not Found!" << endl;
        }
    }
    
    return 0;
}