?	??5Φk`@??5Φk`@!??5Φk`@	&?R?Zl??&?R?Zl??!&?R?Zl??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??5Φk`@???;?^@1eȱ??@A!u;?ʣ?I??~?N@Y.c}???*	 ?rh?%T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK??q?ߕ?!a0)κ?:@)??"???1??????5@:Preprocessing2U
Iterator::Model::ParallelMapV27qr?CQ??!g1???3@)7qr?CQ??1g1???3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\?-??e??!??N?Ȑ=@)VfJ?o	??1@8?o3@:Preprocessing2F
Iterator::Modelz?):?˟?!??1?CC@)??n?????1>?????2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceTƿϸ??!?$?T?C$@)Tƿϸ??1?$?T?C$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?G???\??!-??/?N@)?5?;N?q?1q??@q?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?R?Gn?!??N?kX@)?R?Gn?1??N?kX@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&?R?Zl??I?z??;X@Q?K敏@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???;?^@???;?^@!???;?^@      ??!       "	eȱ??@eȱ??@!eȱ??@*      ??!       2	!u;?ʣ?!u;?ʣ?!!u;?ʣ?:	??~?N@??~?N@!??~?N@B      ??!       J	.c}???.c}???!.c}???R      ??!       Z	.c}???.c}???!.c}???b      ??!       JGPUY&?R?Zl??b q?z??;X@y?K敏@?"<
sequential_14/dense_269/MatMulMatMul?V?????!?V?????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits??D????!bh?꧷??"J
,gradient_tape/sequential_14/dense_269/MatMulMatMuli?? .r??!|?)s3T??0"-
IteratorGetNext/_3_Send?	??`??!???c???"1
Nadam/Nadam/update/addAddV2&??b?v?!?[??????"3
Nadam/Nadam/update/add_2AddV2?@(??u?!???????"3
Nadam/Nadam/update/add_1AddV2????F\q?!9`?MƳ??"9
Nadam/Nadam/update/truediv_3RealDivt?\??Fp?!?.'?2???"3
Nadam/Nadam/update/add_3AddV2X?bn?!?/?C???"<
sequential_14/dense_289/SoftmaxSoftmaxX?bn?!60?>S???Q      Y@Y#%?T?/??ako??@?X@q1$N???W@y*?N???"?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 