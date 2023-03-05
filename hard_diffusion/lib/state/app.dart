
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:hard_diffusion/api/network_service.dart';
import 'package:html/parser.dart';
import 'package:uuid_type/uuid_type.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

final JsonDecoder _decoder = JsonDecoder();

class MyAppState extends ChangeNotifier {
  WebSocketChannel? channel;
  bool webSocketConnected = false;
  bool needsRefresh = false;
  int webSocketReconnectAttempts = 0;
  Map<Uuid, int> taskCurrentStep = {};
  Map<Uuid, int> taskTotalSteps = {};
  Map<Uuid, String> taskPreview = {};
  var ns = NetworkService();
  final inferenceFormKey = GlobalKey<FormState>();
  String prompt = "";
  String negativePrompt = "";
  int seed = 0;
  bool useAdvanced = false;
  bool useMultipleModels = false;
  bool useNsfw = false;
  bool usePreview = false;
  bool useRandomSeed = false;
  int width = 512;
  int height = 512;
  int inferenceSteps = 50;
  double guidanceScale = 7.5;

  void setPrompt(value) {
    prompt = value;
    notifyListeners();
  }

  void setNegativePrompt(value) {
    negativePrompt = value;
    notifyListeners();
  }

  void setUseMultipleModels(value) {
    useMultipleModels = value;
    notifyListeners();
  }

  void setUseNsfw(value) {
    useNsfw = value;
    notifyListeners();
  }

  void setUseAdvanced(value) {
    useAdvanced = value;
    notifyListeners();
  }

  void setUsePreview(value) {
    usePreview = value;
    notifyListeners();
  }

  void setUseRandomSeed(value) {
    useRandomSeed = value;
    notifyListeners();
  }

  void setSeed(value) {
    seed = value;
    notifyListeners();
  }

  void setWidth(value) {
    width = value;
    notifyListeners();
  }

  void setHeight(value) {
    height = value;
    notifyListeners();
  }

  void setInferenceSteps(value) {
    inferenceSteps = value;
    notifyListeners();
  }

  void setGuidanceScale(value) {
    guidanceScale = value;
    notifyListeners();
  }

  void generate() async {
    if (inferenceFormKey.currentState!.validate()) {
      inferenceFormKey.currentState!.save();
      var map = new Map<String, dynamic>();
      map["prompt"] = prompt;
      map["negative_prompt"] = negativePrompt;
      map["use_multiple_models"] = useMultipleModels ? "true" : "";
      map["use_nsfw"] = useNsfw ? "true" : "";
      map["use_preview"] = usePreview ? "true" : "";
      map["use_random_seed"] = useRandomSeed ? "true" : "";
      map["seed"] = seed.toString();
      map["width"] = width.toString();
      map["height"] = height.toString();
      map["inference_steps"] = inferenceSteps.toString();
      map["guidance_scale"] = guidanceScale.toString();
      var response = await ns.get("http://localhost:8000/csrf");
      var document = parse(response);
      var csrf = document.querySelectorAll("input").first.attributes["value"];
      map["csrfmiddlewaretoken"] = csrf;
      ns.headers["X-CSRFToken"] = csrf!;
      response = await ns.post(
        "http://localhost:8000/queue_prompt",
        body: map,
      );
    }
  }

  MyAppState() {
    connect();
  }
  void onMessage(message) {
    var decoded = _decoder.convert(message);
    var decodedMessage = decoded["message"];
    webSocketConnected = true;
    webSocketReconnectAttempts = 0;
    var event = decoded["event"];
    if (event == "image_generated") {
      needsRefresh = true;
    } else if (event == "image_generating") {
      decodedMessage = _decoder.convert(decodedMessage);
      var task_id = Uuid.parse(decodedMessage["task_id"]);
      taskCurrentStep[task_id] = decodedMessage["step"];
      taskTotalSteps[task_id] = decodedMessage["total_steps"];
      if (decodedMessage["image"] != null) {
        taskPreview[task_id] = decodedMessage["image"];
      }
    } else if (event == "image_queued") {
      decodedMessage = Uuid.parse(decodedMessage);
      taskCurrentStep[decodedMessage] = 0;
      taskTotalSteps[decodedMessage] = 0;
      taskPreview[decodedMessage] = "";
      needsRefresh = true;
    } else if (event == "image_errored") {
      needsRefresh = true;
    }
    print(event);
    notifyListeners();
  }

  void onDone() async {
    var delay = 1 + 1 * webSocketReconnectAttempts;
    if (delay > 10) {
      delay = 10;
    }
    print(
        "Done, reconnecting in $delay seconds, attempt $webSocketReconnectAttempts ");
    webSocketConnected = false;
    channel = null;
    await Future.delayed(Duration(seconds: delay));
    connect();
  }

  void onError(error) {
    print(error);
    if (error is WebSocketChannelException) {
      webSocketReconnectAttempts += 1;
    }
  }

  void connect() {
    try {
      channel = WebSocketChannel.connect(
        Uri.parse('ws://localhost:8000/ws/generate/'),
      );
      channel!.stream.listen(onMessage, onDone: onDone, onError: onError);
    } catch (e) {
      print(e);
    }
  }

  void didRefresh() {
    needsRefresh = false;
  }

  void sendMessage(message) {
    if (channel == null) {
      connect();
    }
    if (webSocketConnected && message != null && message.isNotEmpty) {
      channel!.sink.add(message);
    }
  }
}
