#include<iostream>
#include<unordered_map>
#include<vector>
#include<string>
#include<algorithm>
#include<fstream>
#include<set>
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

unordered_map<string,int> get_syntaxs(string input) {
	unordered_map<string,int> res;
	ifstream infile(input);
	string temp;
	int i = 0;
	while(getline(infile,temp))
		res[temp.substr(10)+" "] = i++;
	infile.close();
	return res;
}

vector<string> get_different_class(string input) {
	vector<string> res;
	ifstream infile(input);
	string temp;
	while(getline(infile,temp))
		res.push_back(temp);
	infile.close();
	return res;
}

set<int> class_index(string synsets, string diff_class) {
	unordered_map<string,int> res = get_syntaxs(synsets);
	vector<string> p = get_different_class(diff_class);
	set<int> index;
	for(int i = 0;i < p.size();i++) {
		if(res.find(p[i]) == res.end())
			continue;
		index.insert(res[p[i]]);
	}
	return index;
}

void get_precision(set<int>& index, unordered_map<string,int>& val, unordered_map<string, vector<int>>& res) {
	int one = 0, five = 0, size = 0;
	for(auto k = res.begin();k != res.end();k++) {
		if(val.find(k->first) == val.end())
			continue;
		if(index.find(val[k->first]) != index.end())
			continue;
		size++;
		vector<int> tmp = k->second;
		one += (tmp[0] == val[k->first]);
		for(auto p:tmp) 
			five += (p == val[k->first]);
	}
	cout << "size: " << size << endl;
	cout << one / (size*1.0) << endl;
	cout << five / (size*1.0) << endl;
	cout << endl;
}

int main() {
	int n = 9;
	set<int> index = class_index("synset_words.txt","same_class.txt");
    unordered_map<string, int> val = readVal("val2012.txt");
	unordered_map<string,vector<int>> res;

	res = readResult("result.txt");
    cout << "result: " << endl;
    cout << "size: " << res.size() << endl;
    int one = 0, five = 0;
    for(auto k = res.begin();k != res.end();k++) {
        if(val.find(k->first) == val.end())
            continue;
        vector<int> tmp = k->second;
        one += (tmp[0] == val[k->first]);
        for(auto p:tmp)
            five += (p == val[k->first]);
    }
    cout << one/(res.size()*1.0) << endl;
    cout << five/(res.size()*1.0) << endl;
	cout << "result: "<< endl;
	get_precision(index,val,res);


    return 0;
}

