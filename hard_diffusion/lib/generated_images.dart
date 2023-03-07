import 'dart:async';
import 'package:hard_diffusion/api/images/generated_image.dart';
import 'package:hard_diffusion/exception_indicators/empty_list_indicator.dart';
import 'package:hard_diffusion/exception_indicators/error_indicator.dart';
import 'package:hard_diffusion/image_details.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:infinite_scroll_pagination/infinite_scroll_pagination.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

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
      PagingController(firstPageKey: 1);

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
      final newPage = await fetchPhotos(pageKey, _pageSize);
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

  Widget getImage(item) {
    return ImageDetails(item: item);
  }

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    if (appState.needsRefresh) {
      _pagingController.refresh();
      appState.didRefresh();
    }
    return RefreshIndicator(
      onRefresh: () => Future.sync(
        () => _pagingController.refresh(),
      ),
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
      ),
    );
  }

  @override
  void dispose() {
    _pagingController.dispose();
    super.dispose();
  }
}
