?	???G
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
	2uWv??o@2uWv??o@!2uWv??o@      ??!       "	3???y@3???y@!3???y@*      ??!       2	??S ?g????S ?g??!??S ?g??:	?L2rf@?L2rf@!?L2rf@B      ??!       J	%xC8??%xC8??!%xC8??R      ??!       Z	%xC8??%xC8??!%xC8??b      ??!       JGPUYջ??RG??b q?U?(?X@y?K;n?????"<
sequential_25/dense_500/MatMulMatMulGX?????!GX?????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsp5?y????!?0?<I??"J
,gradient_tape/sequential_25/dense_500/MatMulMatMulT??????!{?P???0"-
IteratorGetNext/_3_Send?+/@???!`??3A??"1
Nadam/Nadam/update/addAddV2??![l?v?!5? ?@??"3
Nadam/Nadam/update/add_2AddV2??![l?v?!??????"9
Nadam/Nadam/update/truediv_3RealDivކh#*Cq?!s???	??"3
Nadam/Nadam/update/add_1AddV2=`???:q?!w?????"1
Nadam/Nadam/update/SqrtSqrtJH?6n?!?GS(7??"3
Nadam/Nadam/update/add_3AddV2JH?6n?!???? ??Q      Y@Y#%?T?/??ako??@?X@q??7???W@y??&????"?

both?Your program is POTENTIALLY input-bound because 96.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 