?	I?s
??b@I?s
??b@!I?s
??b@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-I?s
??b@?b?0m|a@1x? #?@A????C§?I?(???@*	?G?z&X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???O???!???/:?@)V}??b??1~??m?9@:Preprocessing2U
Iterator::Model::ParallelMapV2(?N>=??!??2??{6@)(?N>=??1??2??{6@:Preprocessing2F
Iterator::ModelްmQf???!8?<?D@)?9?S?ɒ?1?U݇e?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?G?`็?!nm????'@)?G?`็?1nm????'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?,??o??!!???Y?4@)??Z	?%??1Դ^?V!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Z|
????!??w??BM@)#??fF?z?1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJΉ=??u?!8?v?X?@)JΉ=??u?18?v?X?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??YAYX@Q?h????@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?0m|a@?b?0m|a@!?b?0m|a@      ??!       "	x? #?@x? #?@!x? #?@*      ??!       2	????C§?????C§?!????C§?:	?(???@?(???@!?(???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??YAYX@y?h????@?"<
sequential_19/dense_374/MatMulMatMul??B?z??!??B?z??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits???o??!8[$& ???"J
,gradient_tape/sequential_19/dense_374/MatMulMatMulX$??Y??!<q??W??0"-
IteratorGetNext/_3_Send*?kYr???!iH?c???"1
Nadam/Nadam/update/addAddV2??5?ݪv?!&?f?j??"3
Nadam/Nadam/update/add_2AddV2?7n??v?!?V):	???"9
Nadam/Nadam/update/truediv_3RealDiv??_?KEq?!?U/?]???"3
Nadam/Nadam/update/add_1AddV2?BFlEq?!???X?Ȳ?"<
sequential_19/dense_394/SoftmaxSoftmax????0p?!?r?
?˳?"7
Nadam/Nadam/update/truedivRealDiv?ށ?0p?!ڐ?j?δ?Q      Y@Y#%?T?/??ako??@?X@qJ??8?jW@y???e?=??"?
both?Your program is POTENTIALLY input-bound because 94.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 