clean:
	rm -f logs/*.log
	rm -f *.pyc
	rm -f WrappedGameCode/*.pyc
	rm -f RawGameCode/*pyc
all:
	./execute.sh