# Compression Questions
- Why are there so few OpenCL Compression algorithms?
- Can we bring OpenCL to more places?

### Primitives

### Run Length Encoding (RLE)
Your most basic but sometimes useful approach.
This says "5e"s to generate "eeeee".
This only reduces your size if you have good clustering.
Clustering can be improved with Barrow's Wheeler Transform (BWT) or others.

### Move to Front (MTF)
You start with a symbol alphabet, "abcdefg..." or [0,1,2,3,4,5,...]
then when you see the symbol you emit the index it currently exists in, then move that symbol to the front of the list after use.
This is a transform that can help clustering and shift your data towards zero if symbols are often repeated next to one another.

### Delta / Relative encoding
If you have the set:
[0,1,2,3,4,5,6,7,...]
there are only unique symbols so it is difficult compress with RLE or MTF.
If we transform this to relative delta between symbols we get:
(first symbol has to be literal or assume start from zero)
[0,1,1,1,1,1,1,1,...]
This transformed set is easily compressible with RLE.

### Huffman coding
We build prefix codes to traverse a tree.
Traditionally this takes two passes.
The first pass we generate the distribution of symbols.
The second pass we construct a tree based on the distribution we found.
This lets us assign the shortest prefix/path to the most frequent symbol.

The downside of this is we need ceil(entropy bits) so we can add a bit per symbol in the worst case.

To solve the two pass problem Dynamic Huffman Trees exist where we change the weights as we read symbols.

### Dense Coding
Build a dictionary of symbols, then emit the index of the symbol we see.
If our symbols are long or occur frequently, we will get compression gains.

If our dictionary has a large index, and words are unique or infrequent, then we won't gain much.

### Transposing our data
For formats like JSON we have "objects"
This interleaves potentially similar fields.
If we have knowledge of our data format, we can transpose these interleaved objects into parallel arrays.

### Lyndon Words
UPDATE: Ok, maybe lyndon words aren't what I thought.
I *thought* I read about them in passing and they were a way to break "banana" up into tokens/codewords of {ba}{na}
We could then use those tokens to build a huffman tree or similar.

I tried digging into wikipedia and white papers and it is pretty heavy stuff about lie groups.

- Am I thinking of something else?
- We want a greedy algorithm to break a stream into long codewords

/*
Lyndon words are a way to generate symbols that can be used to describe a stream.
Instead of having a bit or byte boundary, we can generate lyndon words for a set, then use dense coding or huffman coding to encode a data stream.

- What if we used lyndon words *before* BWT?
  - Lyndon words can be variable length I think? We can probably work around that, but it would be awkward.
  - If we did this, then we'd get good runs of lyndon words, which could give good compression.

- What if we used LZ77 on lyndon words?

- What if we used Lyndon words, then move to front?
*/


### Barrows Wheeler Transform (BWT)
BWT takes an input and creates a matrix from the input vector where each row is a rotated variant.
Once it has constructed the matrix of variants it sorts the rows.

it emits the final column, (and optionally the row index of what the original input is)
This lets us reverse this transformation.
BWT for some reason or another tends to improve symbol grouping.


input:
[1, 2, 3]

BWT Step 1 (make a matrix)
[1, 2, 3]
[3, 1, 2]
[2, 3, 1]

BWT Step 2 (sort matrix by row)
[1, 2 ,3]
[2, 3, 1]
[3, 2, 1]

We would emit
[1, 2, 3] and index 1

(OK, I shifted the wrong direction. Usually you want to select the LAST column, but I shifted the other way, so I want the first column)
We detected the above error because [3,1,1] is missing a 2 so we could never reconstruct [1,2,3]

### What about using Matrices and journaling a determinant reduction into RREF?
Related to BWT, one of the properties of the determinant of a matrix is the area or volume of the parallel-ogram/piped
You can do any linear transform on a row or column and this won't change, though you have to keep a "journal"
e.g. If you have:
[2,4,6]
[4,6,8]
[6,8,10]

we could either start by subtracting rows/columns or dividing them.
If we factors a 2 from each row, we know the area is
(2 * 2 * 2) * [1,2,3]
              [2,3,4]
              [3,4,5]

We could then subtract rows/columns and swap them around.
If we kept a similar journal for these operations, we should end up at or near RREF *and* be invertible with our journal.

- How would we construct the input matrix? scan 32 bytes into each row and make 32 rows?
  - The redundancy would be very sensitive to our scan size or where our "folds" were.
    Depending on the data, 17, 31 or 33 could fold much better than 32.
    We'd need to either test a bunch of block sizes or find a predictor.
      - In cryptopals we fold and look for entropy to search for the footprint left by the key size.
        We can make a pretty good guess for the key size, we could use a similar approach here to guess a good fold size.
      - If all the folds are about the same then what? go to some power of 2?

- How would we store our journal (so our RREF transform was invertable) efficiently? How would we emit the literal?
  - Emitting the literal would be pretty easy if RREF was successful. We'd need 2 numbers. The fold size, and the number of ones on the main diagonal.
  - Ok, so the literal is pretty implicit, what about the operation journal?
    - Make an enum of ops? We only have a few operations (divide/multiply/add/subtract/swap row/swap col) that take 1 or 2 operands.
      - We could potentially ditch divide/subtract and make swap generic (low numbers rows < Dim, high numbers cols > Dim) or something)
    - if each op has a code word and a known number of operands, then we have a state machine. We can encode each op as a codeword with huffman or arithmetic.


### Arithmetic / Range
We divide a number range into weighted groups and then we can encode some span of symbols into a single number.

### LZ77
We have a look ahead buffer (The symbols we want to match) and a search buffer (A pool to match against).
We're trying to map a string into back offsets and run length pairs.
Literals are <0,0, Lit> more or less.

The hope is we end up with a large number of similar tuples that we can then entropy encode.

### "B-Frames" In LZ77?
*Update* LZ77 can *kinda* do do this.
in the string "abcabc..." you can match any (finite) number of "abc"s after the first 3 literals are emitted.
So the encoder can actually have a length run into the future.
E.G.
<0,0,A>
<0,0,B>
<0,0,C>
<3,N,X> # N can be 120 even though there are only 3 symbols in the past.
You just have to be careful to terminate your match one before the end, so your next char is valid.

What LZ77 *can't* is work around "ABC...ABCDABC..." the "D" will end the match group even though another long match group will follow soon.

*sleepy idea* what if / how can we added "fix up frames" or signal this match will likely happen again soon?
I guess a fix up frame could look like "(offset, shift, (insert...))" to say, go back N bytes, shift them over Y, and insert this.
Having the insert be variable length would be tricky. if we only did 1 literal for the insert, the shift size would be implicitly 1.
I'm not sure this operation would save us much in practice though.

How could we estimate some savings?
Look at english text, and find edit distances like this? This would take a while to setup, test, and science.
My guy tells me that you'd need an elegant insertion mechanism at the very least.

ok, so if we have shift size, we implicitly know the number of characters needing to be inserted.
That makes it basically a pascal string.

There are two obvious strategies.
One would be to have the literal as a pascal string.

Another would be to use some delta encoding referencing either the bytes we are copying from or to.
There could be at least two frame types here, the literal pascal string idea, or some delta encoding idea that is somewhat similar
//end update

In LZ77 we only have "P-Frames" using H.264 speak.

Video encoding is about compressing redundancy within one frame (spatial redundancy). Any benefits from image encoding here help here.

Then *also* in compressing redundancy *between* frames (temporal redundancy).
At ~24-120 frames a second, most of the picture is the same from one frame to the next.
(this is done largely with "Motion Estimation" of where blocks/pixels move from one frame to the next)

In video encoding there are 3 frame types (I,P,and B) (P for Previous, B for Bi, I for Independent?)

I-Frames in H.264 which are basically the same as a JPEG.
MJPEG, A precusor, only used I-Frames from what I recall.

MPEG-2 I think came next and introduced "P-Frames"
This was huge because a P-Frame says, "I'm like this last frame except with these changes"
So the bits per P-Frame are pretty low compared to an I-Frame.

H.264 introduced "B-Frames".
These were even more compressed than P-Frames because they could reference both directions, a previous frame *and* a frame to come.

From what I've read, LZ77 and friends only search backwards like P-Frames, and don't have a synchronization mechanism to search forward into the stream.
- Is this where predictive encoding comes in? is that different?

What sort of synchronization mechanism could we use if we had a forward reference?
  - Could we ensure that there were enough literal to decode?
    - So we say "and I'm like this back ref, *plus* 20 characters in the future" could we stall the output stream to resolve that? If we ensured the references were independent?
  - How do we ensure the references are independent and resolvable?
    - Yeah, good question.

### Wavelet trees
https://en.wikipedia.org/wiki/Wavelet_Tree
https://www.sciencedirect.com/science/article/pii/S2213020916302105#bib0030

https://alexbowe.com/wavelet-trees/
http://siganakis.com/challenge-design-a-data-structure-thats-small

Wavelet trees seem to build ontop of:
https://alexbowe.com/rrr/

Wavelet trees *could* help us, but I'm not sure.
They let us answer some things like "does symbol X appear between [a,b]?" "how many times?" "what index?"

I see some conflicting information. Some implementations seem to copy the actual string / substring in each child node.

Other folks generate keys/offsets into the array so they don't have to duplicate it, but they complain about bit banging slowdowns.
lots of popcnt hype, but the symbols are not byte aligned, so they have to mask and shift.

See also:
"Succinct Datastructures"


### Markov Chains


# Algorithms that combine the above

### GZIP / Deflate
RFC 1950 / 1951 / 1952

uses LZ77 and huffman codes for the most part.
The LZ77 window is 32Kb backwards.

#### Multi-stream entropy coding (libpz enhancement)

Standard Deflate feeds the entire LZ77 output (offsets, lengths, and literals
mixed together) into a single Huffman coder. libpz splits this into three
independent byte streams — offsets, lengths, and literals — each with its own
Huffman tree. This exploits the fact that each stream has a tighter symbol
distribution (lower entropy) than the combined stream. On Canterbury + Large
corpus this yields **~16% smaller output** for Deflate and **~18% smaller** for
Lza (which uses Range Coder instead of Huffman), with no speed penalty.
Decompression is also faster (+8% for Deflate) because each stream's shorter
codes decode more quickly.

GZIP has pigz for parallel compression, but decompression is mostly single threaded.

pigz chunks the stream into blocks and compresses each block in parallel.

Decompression *can't* be done in parallel for two main reasons
- We don't know where each block starts and ends, so we don't know where the block's huffman tables start.
  - We can solve this by have a "manifest" file as a sidecar (or maybe a trailer?) telling the decompressor the (relative/delta) offset of each block.
    This would let us work with existing tools, and if we didn't find a manifest, we could decompress single threaded and generate one for the next time.
    Dustin said that trailing data *may* be ignored with warnings, if so, we could tack it on the end with a magic number as the last few bytes, then a reverse offset.
    This would let us have "one file" that would work with existing decompressors, rather than the manifest along side.

- A block can back reference a previous block because of LZ77 by up to 32kb. This creates a data dependency between blocks.
  (We can't resolve the back reference until the previous block is finished, when then stalls the next block..., etc)
  - We could have a special compressor that avoids these data dependencies, but it would likely blow our compression ratio, and we couldn't really decompress existing files in parallel

How often do these back references span blocks in practice?
  - I don't know yet, but I would imagine very often.
  - We'd take a compression ratio hit if we started a clean LZ77 window every block.

### BZIP2
Some CUDA/OpenCL BZIP2 implementations exist.
This is because they are mostly about the Barrow's Wheeler Transform which can be done (and undone) in parallel on blocks.
Doing and undoing the BWT is most of the computation. After that it uses "Move to Front" (I think) with entropy encoding, Arithmetic Encoding (I think).


