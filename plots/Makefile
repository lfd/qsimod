# Latex Makefile using latexmk

.PHONY: all clean

all: plot gen

plot:
	Rscript plot.r

gen:
	bash gen_img.sh

pdf:
	Rscript plot.r pdf

clean:
	rm -rf $(OUTPUT)
