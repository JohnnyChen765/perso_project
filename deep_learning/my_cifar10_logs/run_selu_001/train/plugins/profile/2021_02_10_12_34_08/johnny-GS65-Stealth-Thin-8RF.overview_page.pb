?	a???`@a???`@!a???`@	u???????u???????!u???????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6a???`@]?@??_@1_???݀@A?l??爤?Ih?????@Y?"R?.??*	R???Ky@2F
Iterator::Model????]M??!?1??A}V@)??^??1?e?֖U@:Preprocessing2U
Iterator::Model::ParallelMapV2o???׍?!?{٦e?@)o???׍?1?{٦e?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??W??͔?!8?t?@)?1>?^???1??c?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?^ ??!?^N??h@)?J?*n??1s?.<Bp@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??? !?w?!???????)??? !?w?1???????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}?K??Ϥ?!?s???$@)k*??.?n?1@z??oy??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Y?rLf?!?IM???)??Y?rLf?1?IM???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9t???????I???\EX@Q?uE???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?@??_@]?@??_@!]?@??_@      ??!       "	_???݀@_???݀@!_???݀@*      ??!       2	?l??爤??l??爤?!?l??爤?:	h?????@h?????@!h?????@B      ??!       J	?"R?.???"R?.??!?"R?.??R      ??!       Z	?"R?.???"R?.??!?"R?.??b      ??!       JGPUYt???????b q???\EX@y?uE???@?"<
sequential_26/dense_521/MatMulMatMulp?%M?.??!p?%M?.??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?`y!??!bFq??_??"J
,gradient_tape/sequential_26/dense_521/MatMulMatMul????i<??!?.?.??0"-
IteratorGetNext/_3_SendjF?Fˁ?!?Jߩ???"1
Nadam/Nadam/update/addAddV2?+?=w?!00O]'???"3
Nadam/Nadam/update/add_2AddV2?+?=w?!׊_?g3??"9
Nadam/Nadam/update/truediv_3RealDivX?X?q?!X?x	?L??"/
Nadam/Nadam/update/subSub?`??|p?!f?x?T??"3
Nadam/Nadam/update/add_1AddV2?{???|p?! ???g\??"<
sequential_26/dense_541/SoftmaxSoftmax+?$?+tp?!???A?c??Q      Y@Y#%?T?/??ako??@?X@q$m??}V@y?)?ܛ??"?

both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?90.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 