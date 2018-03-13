#include<iostream>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<fstream>
using namespace std;

int to_int1(string str) {
    int n = str.size(), res = 0;
    int i = 0;
    while(i < n) 
        res = res*10 + (str[i++]-'0');
    return res;
}

vector<int> to_int2(string str) {
    int n = str.size(),i = 1,temp = 0;
    vector<int> res;
    while(i < n-1) {
	while(str[i] == ' ') i++;
	temp = 0;
	while(str[i] != ' ' && str[i] != ']')
	    temp = temp*10 + (str[i++]-'0');
	res.push_back(temp);
    }
    return res;
}

unordered_map<string,int> readVal(string file) {
    ifstream infile(file);
    unordered_map<string,int> res;
    string temp;
    while(getline(infile,temp)) 
        res[temp.substr(0,28)] = to_int1(temp.substr(29));
    infile.close();
    return res;
}

unordered_map<string,vector<int>> readResult(string file) {
    ifstream infile(file);
    string temp;
    unordered_map<string,vector<int>> res;
    vector<string> t;
    while(getline(infile,temp)) {
        res[temp.substr(0,28)] = to_int2(temp.substr(29));
    }
    return res;
}

int main() {
    string infile1 = "val.txt", infile2 = "result.txt";
    unordered_map<string, int> val = readVal(infile1);
    unordered_map<string, vector<int>> res = readResult(infile2);
    int one = 0, five = 0;
    for(auto k = val.begin();k != val.end();k++) {
	if(res.find(k->first) == res.end()) 
	    continue;
	vector<int> tmp = res[k->first];
	one += (tmp[0] == k->second);
	for(auto p:tmp) {
	    five += (p == k->second);
	}
    }
    cout << one/50000.0 << endl;
    cout << five/50000.0 << endl;
    return 0;
}

