?	??s??he@??s??he@!??s??he@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??s??he@2 ?.d@1?V	?S@Ab??vKr??I?\?@*	V-?5S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?FZ*oG??!??kR?>@)t_?lW???1?RMQM9@:Preprocessing2F
Iterator::ModeliR
?????!??6klF@)??@?Α?1?s???6@:Preprocessing2U
Iterator::Model::ParallelMapV2?3?z??!أ?b?66@)?3?z??1أ?b?66@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceR<??k??!'?ʞ?$@)R<??k??1'?ʞ?$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??7?ܘ??!?"??xq3@)?r?9>Z|?1??S"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??/?^|q?!P?bj9@)??/?^|q?1P?bj9@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?	F????!Rt?딓K@))A?G?n?1????yi@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI^0)?mX@Q>?ٚJ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2 ?.d@2 ?.d@!2 ?.d@      ??!       "	?V	?S@?V	?S@!?V	?S@*      ??!       2	b??vKr??b??vKr??!b??vKr??:	?\?@?\?@!?\?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q^0)?mX@y>?ٚJ@?"<
sequential_18/dense_353/MatMulMatMul?qvʐ??!?qvʐ??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsFz??b??!j?;????"-
IteratorGetNext/_3_Send4?-0?!7?F{????"J
,gradient_tape/sequential_18/dense_353/MatMulMatMul?TW??5??!n??????0"3
Nadam/Nadam/update/add_2AddV2?QKLa~v?!?8f$????"1
Nadam/Nadam/update/addAddV2Fs|tplu?!??z?,̰?"3
Nadam/Nadam/update/add_1AddV2m??ߚ#q?!??xGfޱ?"9
Nadam/Nadam/update/truediv_3RealDiv*??4	q?!???????"1
Nadam/Nadam/update/SqrtSqrt?^Z? p?!?o??(???"/
Nadam/Nadam/update/mulMul+`?m?!?'?????Q      Y@Y#%?T?/??ako??@?X@q??T@`*X@y???(i??"?
both?Your program is POTENTIALLY input-bound because 94.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 