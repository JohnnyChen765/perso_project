	???G
Zp@???G
Zp@!???G
Zp@	ջ??RG??ջ??RG??!ջ??RG??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???G
Zp@2uWv??o@13???y@A??S ?g??I?L2rf@Y%xC8??*	???MbZ|@2F
Iterator::Model|?E{????!?׳???V@)	ސFN??1?9(??U@:Preprocessing2U
Iterator::Model::ParallelMapV2*7QKs+??!+???^@)*7QKs+??1+???^@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"?4????!?{?g@)Tol?`??1ABB??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??@J??~?!??????)??@J??~?1??????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB>?٬???!U?̿?@)???im{?1Ʀr~V??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?\??ʾ??!@a?? !@)!>???@p?1鏔?k???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn?8)?{l?!?Bņ??)n?8)?{l?1?Bņ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 96.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ջ??RG??I?U?(?X@Q?K;n????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2uWv??o@2uWv??o@!2uWv??o@      ??!       "	3???y@3???y@!3???y@*      ??!       2	??S ?g????S ?g??!??S ?g??:	?L2rf@?L2rf@!?L2rf@B      ??!       J	%xC8??%xC8??!%xC8??R      ??!       Z	%xC8??%xC8??!%xC8??b      ??!       JGPUYջ??RG??b q?U?(?X@y?K;n????