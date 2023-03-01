import 'dart:async';
import 'package:hard_diffusion/api/images/generated_image.dart';
import 'package:hard_diffusion/exception_indicators/empty_list_indicator.dart';
import 'package:hard_diffusion/exception_indicators/error_indicator.dart';
import 'package:infinite_scroll_pagination/infinite_scroll_pagination.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:cached_network_image/cached_network_image.dart';

/*
class GeneratedImages extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    /*
    var appState = context.watch<MyAppState>();
    var favorites = appState.favorites;
    if (favorites.isEmpty) {
      return Center(
        child: Text('No Images yet'),
      );
    }*/
    return Flexible(
      child: FutureBuilder<List<GeneratedImage>>(
        future: fetchPhotos(http.Client(), 0, 10),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return const Center(
              child: Text('An error has occurred!'),
            );
          } else if (snapshot.hasData) {
            return PhotosList(photos: snapshot.data!);
          } else {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }
        },
      ),
    );
  }
}
class PhotosList extends StatelessWidget {
  const PhotosList({super.key, required this.photos});

  final List<GeneratedImage> photos;

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
      ),
      itemCount: photos.length,
      itemBuilder: (context, index) {
        return Image.network("http://localhost:8000/${photos[index].filename}");
      },
    );
  }
}
*/

class GeneratedImageListView extends StatefulWidget {
  @override
  _GeneratedImageListViewState createState() => _GeneratedImageListViewState();
}

class _GeneratedImageListViewState extends State<GeneratedImageListView> {
  static const _pageSize = 10;

  List<GeneratedImage> _generatedImageList = [];
  final PagingController<int, GeneratedImage> _pagingController =
      PagingController(firstPageKey: 0);

  Future<void> insertGeneratedImageList(List<GeneratedImage> platformList) =>
      Future.microtask(() => _generatedImageList = platformList);

  Future<List<GeneratedImage>> getGeneratedImageList() =>
      Future.microtask(() => _generatedImageList);

  @override
  void didUpdateWidget(GeneratedImageListView oldWidget) {
    // if filters changed...
    // if (oldWidget.listPreferences != widget.listPreferences) {
    //  _pagingController.refresh();
    // }
    super.didUpdateWidget(oldWidget);
  }

  @override
  void initState() {
    _pagingController.addPageRequestListener((pageKey) {
      _fetchPage(pageKey);
    });
    super.initState();
  }

  Future<void> _fetchPage(int pageKey) async {
    try {
      final newPage = await fetchPhotos(http.Client(), pageKey, _pageSize);
      final previouslyFetchedItemsCount =
          // 2
          _pagingController.itemList?.length ?? 0;

      final isLastPage = newPage.isLastPage(previouslyFetchedItemsCount);
      final newItems = newPage.itemList;

      if (isLastPage) {
        // 3
        _pagingController.appendLastPage(newItems);
      } else {
        final nextPageKey = pageKey + 1;
        _pagingController.appendPage(newItems, nextPageKey);
      }
    } catch (error) {
      // 4
      _pagingController.error = error;
    }
  }

  final Image errorImage = Image.network("http://localhost:8000/error.gif");
  final Image paintingImage =
      Image.network("http://localhost:8000/painting.gif");
  Widget getImage(item) {
    if (item.error) {
      return errorImage;
    } else if (item.generatedAt == null) {
      return paintingImage;
    }
    return CachedNetworkImage(
      placeholder: (context, url) => const CircularProgressIndicator(),
      imageUrl: "http://localhost:8000/${item.filename}",
      imageBuilder: (context, imageProvider) => Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: imageProvider,
            fit: BoxFit.fill,
            //colorFilter: ColorFilter.mode(Colors.red, BlendMode.colorBurn)),
          ),
        ),
      ),
      errorWidget: (context, url, error) => errorImage,
    );
  }

  @override
  Widget build(BuildContext context) => RefreshIndicator(
      onRefresh: () => Future.sync(
            () => _pagingController.refresh(),
          ),
      // Don't worry about displaying progress or error indicators on screen; the
      // package takes care of that. If you want to customize them, use the
      // [PagedChildBuilderDelegate] properties.
      child: PagedGridView(
        pagingController: _pagingController,
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
        ),
        builderDelegate: PagedChildBuilderDelegate<GeneratedImage>(
          itemBuilder: (context, item, index) => getImage(item),
          firstPageErrorIndicatorBuilder: (context) => ErrorIndicator(
            error: _pagingController.error,
            onTryAgain: () => _pagingController.refresh(),
          ),
          noItemsFoundIndicatorBuilder: (context) => EmptyListIndicator(),
        ),
        //padding: const EdgeInsets.all(16),
        //separatorBuilder: (context, index) => const SizedBox(
        //  height: 16,
        //),
      ));

  @override
  void dispose() {
    _pagingController.dispose();
    super.dispose();
  }
}
