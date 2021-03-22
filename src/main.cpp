#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <fftw3.h>

#include <math.h>
#include <stdio.h>

#include <QApplication>
#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <QScrollArea>
#include <QVector>
#include <QWidget>
#include <QtGui>

#define REAL 0
#define IMAG 1

/*
 * We aim to detect frequencies in human hearing range which is 20Hz..20kHz. To
 * detect low frequencies chunks must be 1/20sec long as minimum. 1/20 seconds
 * is possible best sampling frequnecy/20, so 48,000/20=2400.
 */

#define HALF_BLOCKSIZE 2400
#define FULL_BLOCKSIZE (HALF_BLOCKSIZE * 2)

class WCwindow : public QMainWindow {
public:
  WCwindow();

};

WCwindow::WCwindow() { setWindowTitle("FFT"); }

QImage Mat2QImage(cv::Mat const &src) {
  cv::Mat temp;

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

  std::cout << "min val: " << minVal << std::endl;
  std::cout << "max val: " << maxVal << std::endl;

  src.convertTo(temp, CV_8UC1, 255.0 / maxVal);

  QImage dest((const unsigned char *)temp.data, temp.cols, temp.rows, temp.step,
              QImage::Format_Grayscale8);
  dest.bits();
  return dest;
}

int main(int argc, char *argv[]) {
  const char *infilename;
  SNDFILE *infile = NULL;
  SF_INFO sfinfo;

  if (argc != 2) {
    infilename = "pulpfiction.wav";
  } else {
    infilename = argv[1];
  }

  std::cout << "Reading audio file '" << infilename << "'" << std::endl;
  if ((infile = sf_open(infilename, SFM_READ, &sfinfo)) == NULL) {
    printf("Not able to open input file %s.\n", infilename);
    puts(sf_strerror(NULL));
    return 1;
  };

  std::cout << "Samplerate " << sfinfo.samplerate << std::endl;
  std::cout << "Frames " << sfinfo.frames << std::endl;
  std::cout << "Channels " << sfinfo.channels << std::endl;
  std::cout << "Format " << sfinfo.format << std::endl;
  std::cout << "Sections " << sfinfo.sections << std::endl;
  std::cout << "Seekable " << sfinfo.seekable << std::endl;

  int subformat = sfinfo.format & SF_FORMAT_SUBMASK;

  switch (subformat) {
  case SF_FORMAT_PCM_S8: //  = 0x0001 Signed 8 bit data
    [[fallthrough]];
  case SF_FORMAT_PCM_16: //  = 0x0002 Signed 16 bit data
    [[fallthrough]];
  case SF_FORMAT_PCM_24: //  = 0x0003 Signed 24 bit data
    [[fallthrough]];
  case SF_FORMAT_PCM_32: //  = 0x0004 Signed 32 bit data
    [[fallthrough]];
  case SF_FORMAT_PCM_U8: //  = 0x0005 Unsigned 8 bit data (WAV and RAW only)
    std::cout << "Integer format (" << subformat << "), good!" << std::endl;
    break;
  case SF_FORMAT_FLOAT: //  = 0x0006, /* 32 bit float data
    [[fallthrough]];
  case SF_FORMAT_DOUBLE: //  = 0x0007, /* 64 bit float data
    std::cout << "Floating point format (" << subformat
              << "), don't know how to work on it!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Processing audio in chunks of " << FULL_BLOCKSIZE << " frames"
            << std::endl;

  // fftw_complex is typedef double fftw_complex[2];
  fftw_complex signal[FULL_BLOCKSIZE];
  fftw_complex result[FULL_BLOCKSIZE];
  /* Prepare FFTW plan */
  fftw_plan plan = fftw_plan_dft_1d(FULL_BLOCKSIZE, signal, result,
                                    FFTW_FORWARD, FFTW_ESTIMATE);

  // Short below matches WAV type 2
  signed short int buf[sfinfo.channels * FULL_BLOCKSIZE];

  std::cout << "allocating memory to hold " << sfinfo.frames << " frames"
            << std::endl;
  printf("Running FFT's on input data\n");
  cv::Mat items(sfinfo.frames / FULL_BLOCKSIZE + 1, HALF_BLOCKSIZE, CV_32FC1,
                cv::Scalar(0));
  int block = 0;
  int readcount;
  // Short below matches WAV type 2
  while ((readcount = sf_readf_short(infile, buf, FULL_BLOCKSIZE)) > 0) {
    /* Detecting end of file or other errors while reading file */
    if (readcount != FULL_BLOCKSIZE) {
      std::cout << "sf_readf_short got " << readcount << " after reading "
                << block << " full blocks. Possibly end of file" << std::endl;
      memset(signal, 0, sizeof(signal));
      std::cout << "memsetting " << FULL_BLOCKSIZE << " * sizeof(double) "
                << sizeof(double) << " * 2 = " << sizeof(signal)
                << " bytes to 0" << std::endl;
    }
    /*
     * Loop over input data, calculate mono value and input data into FFT
     */
    for (int i = 0; i < readcount; i++) {
      double real = 0.0;
      for (int j = 0; j < sfinfo.channels; j++) {
        real += (double)(buf[i * sfinfo.channels + j]);
        // std::cout << "+" << i << " " << buf[i * sfinfo.channels + j] << " "
        // << real <<std::endl;
      }
      signal[i][REAL] = real / (double)sfinfo.channels;
      signal[i][IMAG] = 0.0;
      // std::cout << "=" << i << " " << real << " " <<  signal[i][REAL] <<
      // std::endl;
    }
    fftw_execute(plan);
    /*
     * For every FULL_BLOCKSIZE bytes input we receive FULL_BLOCKSIZE bytes of
     * FFT output. But only first half of results are usable. Second half of
     * results is mirror of first half, so we omit it
     */
    for (int i = 0; i < HALF_BLOCKSIZE; i++) {
      /*
       * Formula with logarithm and powers are needed to get magnitude of
       * signal. At moment we do not need it but phase can be also computer with
       * formula Phase = arctan(Imaginary(F)/Real(F))
       */
      double tmp = sqrt(pow(result[i][REAL], 2.0) + pow(result[i][IMAG], 2.0));
      // log10(sqrt(pow(result[i][REAL], 2.0) + pow(result[i][IMAG], 2.0)));
      /* Next keeps it safe from float value "inf" which ruins the game */
      tmp = isinf(tmp) ? 0.0 : tmp;
      items.at<float>(block, i) = (float)tmp;

      if (block == 1000) {
        std::cout << i << " " << tmp << std::endl;
      }
    }
    block++;
  };
#if 0
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  minMaxLoc(items, &minVal, &maxVal, &minLoc, &maxLoc);

  std::cout << "min val: " << minVal << std::endl;
  std::cout << "max val: " << maxVal << std::endl;

 // items *= 255.0 / (float)maxVal;
#endif
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  bool r = false;

  try {
    r = imwrite("items.png", items, compression_params);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (r)
    printf("Saved PNG file\n");
  else
    printf("ERROR: Can't save PNG file.\n");

  fftw_destroy_plan(plan);

  QApplication a(argc, argv);
  QWidget W1;
  QLabel imlab1(&W1);

  QImage qimg = Mat2QImage(items);

  imlab1.setPixmap(QPixmap::fromImage(qimg));
  W1.setFixedSize(qimg.size());
  WCwindow win;

  QScrollArea *scrollArea = new QScrollArea;
  scrollArea->setWidget(&W1);
  scrollArea->setWidgetResizable(true);

  win.setCentralWidget(scrollArea);

  win.show();
  return a.exec();
}
