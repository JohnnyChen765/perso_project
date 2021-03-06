?	?g\WU`@?g\WU`@!?g\WU`@	o?!&??o?!&??!o?!&??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?g\WU`@?=&R?D^@1???G?@AS?
cA??I?A|`?@Y(?N>=???*	??~j??R@2F
Iterator::Model?"M?<??!??D?tF@)?GĔH??1?b=\??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat |(ђǓ?!???[??9@)!>???@??1???DT-5@:Preprocessing2U
Iterator::Model::ParallelMapV2??R?r/??!???,?5@)??R?r/??1???,?5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?U?????!"k?%;?8@) ??*Q???1?=XD?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice6Y???}?!??w?1m#@)6Y???}?1??w?1m#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!;oc?#??!N?p?G?K@)?"??l?1z?֗?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???k?6l?!jɈ]ha@)???k?6l?1jɈ]ha@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9n?!&??IU??P:X@Q?,?ҌI@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?=&R?D^@?=&R?D^@!?=&R?D^@      ??!       "	???G?@???G?@!???G?@*      ??!       2	S?
cA??S?
cA??!S?
cA??:	?A|`?@?A|`?@!?A|`?@B      ??!       J	(?N>=???(?N>=???!(?N>=???R      ??!       Z	(?N>=???(?N>=???!(?N>=???b      ??!       JGPUYn?!&??b qU??P:X@y?,?ҌI@?"<
sequential_22/dense_437/MatMulMatMul?8ɷ????!?8ɷ????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits'?|?:??!??C?,_??"-
IteratorGetNext/_3_SendC??I?ق?!??????"J
,gradient_tape/sequential_22/dense_437/MatMulMatMul?"??
???!?՝?ݵ??0"1
Nadam/Nadam/update/addAddV2??ēe4v?!<tDj|??"3
Nadam/Nadam/update/add_2AddV2??ēe4v?!{?G{{???"3
Nadam/Nadam/update/add_1AddV2?ߢs??q?!w??????"9
Nadam/Nadam/update/truediv_3RealDiv X????p?!?<?
?ϲ?"1
Nadam/Nadam/update/SqrtSqrt?#Ƿ?o?!EZ?Ȉͳ?"9
Nadam/Nadam/update/truediv_2RealDiv?#Ƿ?o?!?w??N˴?Q      Y@Y#%?T?/??ako??@?X@q?|X?U@y&??C???"?
both?Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?84.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 