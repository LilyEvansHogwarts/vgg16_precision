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
    while(getline(infile,temp)) {
        res[temp.substr(0,28)] = to_int1(temp.substr(29));
    }
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
	int n = 9;
	vector<unordered_map<string, vector<int>>> res;
	for(int i = 0;i < n;i++) 
		res.push_back(readResult("t" + to_string(i+1) + ".txt"));
    unordered_map<string, int> val = readVal("val.txt");
	vector<int> one(n,0), five(n,0);
	int total_one = 0, total_five, total_size = 0;
	for(int i = 0;i < n;i++) {
		for(auto k = res[i].begin();k != res[i].end();k++) {
			if(val.find(k->first) == val.end())
				continue;
			vector<int> tmp = res[i][k->first];
			one[i] += (tmp[4] == val[k->first]);
			for(auto p:tmp)
				five[i] += (p == val[k->first]);
		}
		total_one += one[i];
		total_five += five[i];
		total_size += res[i].size();
		cout << "package " << i+1 << ":" << endl;
		cout << one[i]/(res[i].size() * 1.0) << endl;
		cout << five[i]/(res[i].size() * 1.0) << endl;
	}
	cout << "total:" << endl;
	cout << total_one/(total_size * 1.0) << endl;
	cout << total_five/(total_size * 1.0) << endl;

    return 0;
}

