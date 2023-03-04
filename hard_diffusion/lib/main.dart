// import 'dart:convert';

import 'dart:convert';

import 'package:flutter/material.dart';
//import 'package:flutter_localizations/flutter_localizations.dart';
//import 'package:http/http.dart' as http;
import 'package:flutter_gen/gen_l10n/app_localizations.dart';
import 'package:english_words/english_words.dart';
import 'package:hard_diffusion/generate.dart';
import 'package:hard_diffusion/vertical_separator.dart';
import 'package:provider/provider.dart';
import 'package:uuid_type/uuid_type.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  runApp(const MyApp());
}

/*
Example code, from https://flutter.dev/docs/cookbook/networking/fetch-data
Future<Album> fetchAlbum() async {
  final response = await http
      .get(Uri.parse('https://jsonplaceholder.typicode.com/albums/1'));
  /*
        // Send authorization headers to the backend.
  headers: {
    HttpHeaders.authorizationHeader: 'Basic your_api_token_here',
  },
      */
  if (response.statusCode == 200) {
    // If the server did return a 200 OK response,
    // then parse the JSON.
    return Album.fromJson(jsonDecode(response.body));
  } else {
    // If the server did not return a 200 OK response,
    // then throw an exception.
    throw Exception('Failed to load album');
  }
}

class Album {
  final int userId;
  final int id;
  final String title;

  const Album({
    required this.userId,
    required this.id,
    required this.title,
  });

  factory Album.fromJson(Map<String, dynamic> json) {
    return Album(
      userId: json['userId'],
      id: json['id'],
      title: json['title'],
    );
  }
}*/

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => MyAppState(),
      child: MaterialApp(
          onGenerateTitle: (context) =>
              AppLocalizations.of(context)!.helloWorld,
          title: 'Hard Diffusion',
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          theme: ThemeData(
            // This is the theme of your application.
            //
            // Try running your application with "flutter run". You'll see the
            // application has a blue toolbar. Then, without quitting the app, try
            // changing the primarySwatch below to Colors.green and then invoke
            // "hot reload" (press "r" in the console where you ran "flutter run",
            // or simply save your changes to "hot reload" in a Flutter IDE).
            // Notice that the counter didn't reset back to zero; the application
            // is not restarted.
            primarySwatch: Colors.lightGreen,
            //primarySwatch: null,
            //colorSchemeSeed: Colors.deepOrange,
            //colorScheme: ColorScheme.fromSeed(seedColor: Colors.grey),
          ),
          home: MyHomePage()
          // const MyHomePage(title: 'Hard Diffusion'),
          ),
    );
  }
}

final JsonDecoder _decoder = JsonDecoder();

class MyAppState extends ChangeNotifier {
  var current = WordPair.random();
  WebSocketChannel? channel;
  bool webSocketConnected = false;
  bool needsRefresh = false;
  int webSocketReconnectAttempts = 0;
  Map<Uuid, int> taskCurrentStep = {};
  Map<Uuid, int> taskTotalSteps = {};
  Map<Uuid, String> taskPreview = {};

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

  void getNext() {
    current = WordPair.random();
    notifyListeners();
  }

  void didRefresh() {
    needsRefresh = false;
  }

  var favorites = <WordPair>{};

  void toggleFavorite() {
    if (favorites.contains(current)) {
      favorites.remove(current);
    } else {
      favorites.add(current);
    }
    notifyListeners();
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

class MyHomePage extends StatefulWidget {
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var selectedIndex = 0;

  /*
  late Future<Album> futureAlbum;
  @override
  void initState() {
    super.initState();
    futureAlbum = fetchAlbum();
  }
  */

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var channel = appState.channel;
    Widget page = GeneratePage();
    switch (selectedIndex) {
      /*case 0:
        page = GeneratorPage();
        break;
      case 1:
        page = FavoritesPage();
        break;*/
      case 0:
        page = GeneratePage();
        break;
      default:
        print("Nope");
        break;
      //throw UnimplementedError("no widget for $selectedIndex");
    }
    return LayoutBuilder(builder: (context, constraints) {
      return Scaffold(
        body: Row(
          children: [
            SafeArea(
              child: NavigationRail(
                //extended: constraints.maxWidth >= 800,
                destinations: [
                  NavigationRailDestination(
                    icon: Icon(Icons.home),
                    label: Text('Generate'),
                  ),
                  NavigationRailDestination(
                    icon: Icon(Icons.settings),
                    label: Text('Settings'),
                  ), /*
                    NavigationRailDestination(
                      icon: Icon(Icons.favorite),
                      label: Text('Favorites'),
                    ),*/
                ],
                selectedIndex: selectedIndex,
                onDestinationSelected: (value) {
                  print('selected: $value');
                  setState(() {
                    selectedIndex = value;
                  });
                },
              ),
            ),
            VerticalSeparator(),
            /*
              FutureBuilder<Album>(
                future: futureAlbum,
                builder: (context, snapshot) {
                  if (snapshot.hasData) {
                    return Text(snapshot.data!.title);
                  } else if (snapshot.hasError) {
                    return Text('${snapshot.error}');
                  }
        
                  // By default, show a loading spinner.
                  return const CircularProgressIndicator();
                },
              ),*/
            Expanded(
              child: Container(
                //color: Theme.of(context).colorScheme.primaryContainer,
                child: page,
              ),
            ),
          ],
        ),
      );
    });
  }
}

class GeneratorPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var pair = appState.current;

    IconData icon;
    if (appState.favorites.contains(pair)) {
      icon = Icons.favorite;
    } else {
      icon = Icons.favorite_border;
    }

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          BigCard(pair: pair),
          SizedBox(height: 10),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              ElevatedButton.icon(
                onPressed: () {
                  appState.toggleFavorite();
                },
                icon: Icon(icon),
                label: Text('Like'),
              ),
              SizedBox(width: 10),
              ElevatedButton(
                onPressed: () {
                  appState.getNext();
                },
                child: Text('Next'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class BigCard extends StatelessWidget {
  const BigCard({
    super.key,
    required this.pair,
  });

  final WordPair pair;

  @override
  Widget build(BuildContext context) {
    var theme = Theme.of(context);
    var style = theme.textTheme.displayMedium!
        .copyWith(color: theme.colorScheme.onPrimary);

    return Card(
      elevation: 5,
      color: theme.colorScheme.primary, //Color(0xFFA0FF00), //Color.fromRGBO(
      // 0, 255, 0, 1.0), //Colors.amber, //theme.colorScheme.primary,
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Text(
          pair.asPascalCase /*pair.asLowerCase*/,
          style: style,
          // For screenreadeers if text was lower case.
          semanticsLabel: pair.asPascalCase,
        ),
      ),
    );
  }
}

/*
class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.

    var localization = AppLocalizations.of(context)!;
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          tooltip: MaterialLocalizations.of(context).openAppDrawerTooltip,
          icon: const Icon(Icons.menu),
          onPressed: () {
            Scaffold.of(context).openDrawer();
          },
        ),
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(AppLocalizations.of(context)!.helloWorld),
        actions: [
          IconButton(
            tooltip: "favorite", //localization.starterAppTooltipFavorite,
            icon: const Icon(
              Icons.favorite,
            ),
            onPressed: () {},
          ),
          IconButton(
            tooltip: "search", //localization.starterAppTooltipSearch,
            icon: const Icon(
              Icons.search,
            ),
            onPressed: () {},
          ),
          PopupMenuButton<Text>(
            itemBuilder: (context) {
              return [
                PopupMenuItem(
                  child: Text(
                    "first", //localization.demoNavigationRailFirst,
                  ),
                ),
                PopupMenuItem(
                  child: Text(
                    "second", //localization.demoNavigationRailSecond,
                  ),
                ),
                PopupMenuItem(
                  child: Text(
                    "third", //localization.demoNavigationRailThird,
                  ),
                ),
              ];
            },
          )
        ],
      ),
      body: Center(
          // Center is a layout widget. It takes a single child and positions it
          // in the middle of the parent.
          child: Row(children: [
        Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Invoke "debug painting" (press "p" in the console, choose the
          // "Toggle Debug Paint" action from the Flutter Inspector in Android
          // Studio, or the "Toggle Debug Paint" command in Visual Studio Code)
          // to see the wireframe for each widget.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
        Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ])),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
*/