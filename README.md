BGL
---

A library wrapping boost graph to specifically handle atom-mapping corner cases in single topology FEP. Under the hood this calls into `boost::graph::mcgregor_common_subgraph` routines with a special callback. 

Build
-----

CMake installation instructions

```
mkdir build
cd build && cmake ../
make
```

License
-------
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.