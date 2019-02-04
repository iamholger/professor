// Test for Ipol class

#include "Professor/Ipol.h"
#include <iostream>


int main() {

  using namespace std;

  const vector<double> anchor1{0,0}, anchor2{0,1}, anchor3{0,2};
  const vector< vector<double> > points {anchor1, anchor2, anchor3};
  const vector<double> vals{0, 1, 2};
  const vector<double> point{0.0, 0.5};
  const vector< string > names{"MEG", "CHRIS"};

  Professor::ParamPoints pp(points);

  pp.setNames(names);

  ////cout << "Require " << calcnumCoeffs(2,1) << endl;
  //Professor::Ipol ip1(pp, vals, 1);
  //cout << ip1.value(anchor1) << endl;
  //cout << ip1.value(point) << endl;
  //cout << ip1.toString() << endl;
  //cout << ip1.toString("Crazy") << endl;

  //Professor::Ipol ip2("Test: 2 1 1.11022e-16 0 1");
  //cout << ip2.value(point) << endl;
  //cout << ip2.toString() << endl;

  //Professor::Ipol ip3("2 1 1.11022e-16 0 1");
  //cout << ip3.value(point) << endl;
  //cout << ip3.toString("Awesome") << endl;

  //cout << "\n --- what follows is an example profDF output --- \n\n";

  //// ProfDF stuff
  //cout << pp.toString("This is some description");
  //cout << ip1.toString("Crazy") << endl;
  //cout << ip2.toString("Splendid") << endl;
  //cout << ip3.toString("Awesome") << endl;

  return 0;
}
