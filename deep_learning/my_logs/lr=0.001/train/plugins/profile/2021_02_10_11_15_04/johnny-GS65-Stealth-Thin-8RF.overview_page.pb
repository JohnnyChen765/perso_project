?	?il???a@?il???a@!?il???a@	:??V)R?:??V)R?!:??V)R?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?il???a@?-$`?`@1??^[@A?????K??I??)???@Y??.??Y?*	?&1?DP@2U
Iterator::Model::ParallelMapV29?]????!_??rSd:@)9?]????1_??rSd:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatƤ???!v?Q???=@)?w?-;ď?1G?5?7@:Preprocessing2F
Iterator::Model!?> ?M??!I???<E@)?m½2o??13?.?^0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???͋?!???n?Q'@)???͋?1???n?Q'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????U???!@??)?5@)a?$??z?1?U?V? $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??k*??!?Kr?&?L@)????Mbp?1??G#x?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoreQ?E?o?!?!?E@)eQ?E?o?1?!?E@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9:??V)R?Iok??5eX@Qg3??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-$`?`@?-$`?`@!?-$`?`@      ??!       "	??^[@??^[@!??^[@*      ??!       2	?????K???????K??!?????K??:	??)???@??)???@!??)???@B      ??!       J	??.??Y???.??Y?!??.??Y?R      ??!       Z	??.??Y???.??Y?!??.??Y?b      ??!       JGPUY:??V)R?b qok??5eX@yg3??V@?"<
sequential_20/dense_395/MatMulMatMulr?Fޢ???!r?Fޢ???0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsIǑ0p??!???]*??"-
IteratorGetNext/_3_Send?t?2????!??R`???"J
,gradient_tape/sequential_20/dense_395/MatMulMatMul???*Ɂ?!?%?Ɋ??0"3
Nadam/Nadam/update/add_2AddV2?@?O4?y?!???4ï?"1
Nadam/Nadam/update/addAddV2?KÎ5?x?!??fsj??"9
Nadam/Nadam/update/truediv_3RealDiv]I??s?!*?'4
???"1
Nadam/Nadam/update/SqrtSqrt?TNK?er?!ve?iʳ?"3
Nadam/Nadam/update/add_1AddV2?TNK?er?!?J??????"3
Nadam/Nadam/update/add_3AddV2o?=
=,q?!?#5????Q      Y@Y#%?T?/??ako??@?X@qq????W@y??ƃ,*??"?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 