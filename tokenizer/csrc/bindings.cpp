#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kmer.h"

namespace py = pybind11;

PYBIND11_MODULE(kmer_c, m) {
  py::class_<KMerTokenizer>(m, "KMerTokenizer")
    .def(py::init<int>())
    .def("tokenize_sequence", &KMerTokenizer::tokenize_sequence)
    .def("encode", &KMerTokenizer::encode)
    .def("decode", &KMerTokenizer::decode)
    .def("set_vocab", &KMerTokenizer::set_vocab)
    .def("get_vocab", &KMerTokenizer::get_vocab);
}