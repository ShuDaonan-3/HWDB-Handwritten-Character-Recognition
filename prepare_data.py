from project import ROOT, ARCHIVE, load

if __name__ == '__main__':
    prepare = load('prepare', ARCHIVE/'src/prepare_data.py')
    prepare.ROOT = ROOT
    prepare.main()
