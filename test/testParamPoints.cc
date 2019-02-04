/// Test for ParamPoints class

#include "Professor/ParamPoints.h"

int main(int argc, char** argv) {

  using namespace std;

  const vector<double> anchor1{0.5449,0.6028}, anchor2{0.3834,0.9637}, anchor3{0.5289,0.7917};
  const vector< vector<double> > points{anchor1,anchor2,anchor3};
  const vector< string > names{"STEWIE", "BRIAN"};

  // Test constructor
  Professor::ParamPoints pp(points);
  pp.printMeta();
  pp.printPoints();

  // Test reset and alternative assembly mechanism
  pp.reset();
  pp.addParamPoint(anchor3);
  pp.addParamPoint(anchor2);
  pp.addParamPoint(anchor1);
  pp.printMeta();
  pp.printPoints();
  pp.setNames(names);

  // The following is suitable for storage to disk
  cout << pp.toString();

  return 0;
}
